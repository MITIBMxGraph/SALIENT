from abc import abstractmethod
from typing import Callable, List, Mapping, Type, Iterable
from ogb.nodeproppred import Evaluator
from pathlib import Path
from tqdm import tqdm
import time
import torch

import importlib
if importlib.util.find_spec("torch_geometric.loader") is not None:
    import torch_geometric.loader
    if hasattr(torch_geometric.loader, "NeighborSampler"):
        from torch_geometric.loader import NeighborSampler
    else:
        from torch_geometric.data import NeighborSampler
else:
    from torch_geometric.data import NeighborSampler

import torch.distributed as dist

from ..dataset import FastDataset
from fast_trainer.utils import Timer, CUDAAggregateTimer
from fast_trainer.utils import append_runtime_stats, start_runtime_stats_epoch
from fast_trainer.samplers import *
from fast_trainer.transferers import *
from fast_trainer.concepts import TrainImpl
from fast_trainer import train, test


class BaseDriver:
    devices: List[torch.device]
    dataset: FastDataset
    lr: float
    train_loader: ABCNeighborSampler
    train_transferer: Type[DeviceIterator]
    test_transferer: Type[DeviceIterator]
    train_impl: TrainImpl
    train_max_num_batches: int
    model: torch.nn.Module
    make_subgraph_loader: Callable[[torch.Tensor], Iterable[PreparedBatch]]
    evaluator: Evaluator
    log_file: Path

    def __init__(self, args, devices: List[torch.device],
                 dataset: FastDataset, model_type: Type[torch.nn.Module]):
        assert torch.cuda.is_available()

        self.args = args
        self.devices = devices
        self.dataset = dataset
        self.model_type = model_type
        self.lr = args.lr
        self.log_file = Path(args.log_file)
        self.logs = []
        self.firstRun = True
        self.TRIAL_NUM = 0
        #self.dist_sampler = self.get_sampler(1000)
        assert len(self.devices) > 0

        if args.train_type == 'serial' and len(self.devices) > 1:
            raise ValueError('Cannot serial train with more than one device.')

        # TODO: Add 1D version of serial_idx kernel
        cfg = FastSamplerConfig(
            x=self.dataset.x, y=self.dataset.y.unsqueeze(-1),
            rowptr=self.dataset.rowptr, col=self.dataset.col,
            idx=self.dataset.split_idx['train'],
            batch_size=args.train_batch_size, sizes=args.train_fanouts,
            skip_nonfull_batch=False, pin_memory=True
        )

        self.train_max_num_batches = min(args.train_max_num_batches,
                                         cfg.get_num_batches())

        def make_loader(sampler, cfg: FastSamplerConfig):
            kwargs = dict()
            if sampler == 'NeighborSampler' and self.args.one_node_ddp:
                kwargs = dict(sampler=self.get_sampler(self.TRIAL_NUM*1000 +
                                                       self.global_rank),
                              persistent_workers=True)
            return {
                'FastPreSampler': lambda: FastPreSampler(cfg),
                'FastSampler': lambda: FastSampler(
                    args.num_workers, self.train_max_num_batches, cfg),
                'NeighborSampler': lambda: NeighborSampler(
                    self.dataset.adj_t(), node_idx=cfg.idx,
                    batch_size=cfg.batch_size, sizes=cfg.sizes,
                    num_workers=args.num_workers, pin_memory=True, **kwargs)
            }[sampler]()

        self.train_loader = make_loader(args.train_sampler, cfg)

        self.train_transferer = DevicePrefetcher if args.train_prefetch \
            else DeviceTransferer
        self.test_transferer = DevicePrefetcher if args.test_prefetch \
            else DeviceTransferer

        self.train_impl = {'dp': train.data_parallel_train,
                           'serial': train.serial_train}[args.train_type]

        self.model = self.model_type(
            self.dataset.num_features, args.hidden_features,
            self.dataset.num_classes,
            num_layers=args.num_layers).to(self.main_device)
        self.model_noddp = self.model_type(
            self.dataset.num_features, args.hidden_features,
            self.dataset.num_classes,
            num_layers=args.num_layers).to(self.main_device)

        self.idx_arange = torch.arange(self.dataset.y.numel())

        self.evaluator = Evaluator(name=args.dataset_name)

        self.reset()

    def __del__(self):
        if len(self.logs) > 0:
            raise RuntimeError('Had unflushed logs when deleting BaseDriver')
        # NOTE: Cannot always flush logs for the user.
        # It might be impossible if __del__ is called during
        # the shutdown phase of the interpreter...

        # self.flush_logs()

    def _reset_model(self):
        self.model.reset_parameters()
        #print("Reset model")

    def _reset_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        #print("Reset optimizer")

    def reset(self):
        self._reset_model()
        self._reset_optimizer()
        self.TRIAL_NUM += 1

    @property
    def my_name(self) -> str:
        return self.args.job_name

    @property
    @abstractmethod
    def is_main_proc(self) -> bool:
        ...

    @property
    def main_device(self) -> torch.device:
        return self.devices[0]

    def get_idx_test(self) -> None:
        return self.dataset.split_idx['test']

    def make_train_devit(self) -> DeviceIterator:
        return self.train_transferer(self.devices, iter(self.train_loader))

    def log(self, t) -> None:
        self.logs.append(t)
        if self.is_main_proc and self.args.verbose:
            print(str(t))

    def flush_logs(self) -> None:
        if len(self.logs) == 0:
            return

        with self.log_file.open('a') as f:
            f.writelines(repr(item) + '\n' for item in self.logs)
        self.logs = []

    def train(self, epochs) -> None:
        self.model.train()
        if self.args.model_name.lower() == "sageresinception" or \
           self.args.use_lrs:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer, factor=0.8,
                patience=self.args.patience, verbose=True)
        else:
            lr_scheduler = None

        def record_sampler_init_time(x):
            self.log(x)
            # append_runtime_stats("Sampler init", x.nanos/1000000)

        for epoch in epochs:
            start_runtime_stats_epoch()
            ctimer_preamble = CUDAAggregateTimer("Preamble")
            ctimer_preamble.start()

            with Timer((f'Train epoch {epoch}', 'Preamble'),
                       record_sampler_init_time):
                runtime_stats_cuda.start_region("total")
                runtime_stats_cuda.start_region(
                    "sampling", runtime_stats_cuda.get_last_event())
                if self.args.train_sampler == 'NeighborSampler':
                    self.train_loader.node_idx = self.get_idx(epoch)
                    devit = self.train_loader
                else:
                    self.train_loader.idx = self.get_idx(epoch)
                    devit = self.make_train_devit()
                runtime_stats_cuda.end_region("sampling")
                runtime_stats_cuda.end_region(
                    "total", runtime_stats_cuda.get_last_event())
            ctimer_preamble.end()
            # append_runtime_stats("Sampler init", ctimer_preamble.report())

            if self.args.train_sampler != 'NeighborSampler' and \
               isinstance(devit.it, FastSamplerIter):
                self.log((epoch, devit.it.get_stats()))
                append_runtime_stats("Sampler init", devit.it.get_stats(
                ).total_blocked_dur.total_seconds() * 1000)
            if self.is_main_proc:
                if self.args.train_sampler == 'NeighborSampler':
                    pbar = tqdm(total=self.train_loader.node_idx.numel())
                else:
                    pbar = tqdm(total=self.train_loader.idx.numel())
                pbar.set_description(f'Train epoch {epoch}')

            def cb(inputs, results):
                if self.is_main_proc:
                    pbar.update(sum(batch.batch_size for batch in inputs))

            def cb_NS(inputs, results):
                if self.is_main_proc:
                    pbar.update(sum(bs[0] for bs in inputs))

            def log_total_compute_time(x):
                append_runtime_stats("total", x.nanos/1000000)
                self.log(x)

            with Timer((f'Train epoch {epoch}', 'GPU compute'),
                       log_total_compute_time) as timer:
                if self.args.train_sampler == 'NeighborSampler':
                    self.train_impl(self.model, train.barebones_train_core,
                                    devit, self.optimizer, lr_scheduler,
                                    cb_NS, dataset=self.dataset,
                                    devices=self.devices)
                else:
                    self.train_impl(self.model, train.barebones_train_core,
                                    devit, self.optimizer, lr_scheduler,
                                    cb, dataset=None, devices=None)

                # Barrier is not needed for correctness. I'm also not
                # sure it is needed for accurate timing either because
                # of synchronization in DDP model. In any case,
                # including it here to make sure there is a
                # synchronization point inside the compute region.
                if dist.is_initialized():
                    dist.barrier()
                timer.stop()
                if self.is_main_proc:
                    if self.args.train_sampler != 'NeighborSampler' and \
                            isinstance(devit.it, FastSamplerIter):

                        self.log((epoch, devit.it.get_stats()))
                        # append runtime stats. Convert units to milliseconds
                        append_runtime_stats("Sampling block time",
                                             devit.it.get_stats(
                        ).total_blocked_dur.total_seconds()*1000)
                    pbar.close()
                    del pbar

    def test(self, sets=None) -> Mapping[str, float]:
        # if self.is_main_proc:
        #     print()

        if self.args.test_type == 'layerwise':
            results = self.layerwise_test(sets=sets)
        elif self.args.test_type == 'batchwise':
            results = self.batchwise_test(sets=sets)
        else:
            raise ValueError('unknown test_type')

        return results

    @torch.no_grad()
    def layerwise_test(self, sets=None) -> Mapping[str, float]:
        self.model.eval()

        if sets is None:
            sets = self.dataset.split_idx

        def make_subgraph_iter(x):
            cfg = FastSamplerConfig(
                x=x, y=None,
                rowptr=self.dataset.rowptr, col=self.dataset.col,
                idx=self.idx_arange,
                batch_size=self.args.test_batch_size, sizes=[-1],
                skip_nonfull_batch=False, pin_memory=True
            )
            loader = FastSampler(self.args.num_workers,
                                 self.args.test_max_num_batches, cfg)
            return DeviceTransferer([self.main_device], iter(loader))

        if self.is_main_proc:
            out = self.model.module.inference(self.dataset.x,
                                              self.main_device,
                                              make_subgraph_iter)
            y_true = self.dataset.y.unsqueeze(-1)
            y_pred = out.argmax(dim=-1, keepdim=True)
            ret = [{name:
                    self.evaluator.eval({
                        'y_true': y_true[self.dataset.split_idx[name]],
                        'y_pred': y_pred[self.dataset.split_idx[name]],
                    })['acc']
                    for name in sets}]
        else:
            ret = [{name: 0 for name in sets}]

        if dist.is_initialized():
            dist.broadcast_object_list(ret, src=0)

        return ret[0]

    @torch.no_grad()
    def batchwise_test(self, sets=None) -> Mapping[str, float]:
        self.model.eval()

        if sets is None:
            sets = self.dataset.split_idx

        results = {}

        for name in sets:
            with Timer((name, 'Preamble'), self.log):
                local_fanouts = self.args.batchwise_test_fanouts
                local_batchsize = self.args.test_batch_size

                if name == 'test':
                    local_fanouts = self.args.final_test_fanouts
                    local_batchsize = self.args.final_test_batchsize

                cfg = FastSamplerConfig(
                    x=self.dataset.x, y=self.dataset.y.unsqueeze(-1),
                    rowptr=self.dataset.rowptr, col=self.dataset.col,
                    idx=self.get_idx_test(name),
                    batch_size=local_batchsize,
                    sizes=local_fanouts,
                    skip_nonfull_batch=False, pin_memory=True
                )
                loader = FastSampler(self.args.num_workers,
                                     self.args.test_max_num_batches, cfg)
                devit = self.test_transferer([self.main_device], iter(loader))

            if self.is_main_proc:
                pbar = tqdm(total=cfg.idx.numel())
                if not dist.is_initialized():
                    pbar.set_description(f'{name} (one proc)')
                else:
                    pbar.set_description(
                        f'{name} (multi proc, showing main proc)')

            def cb(batch):
                if self.is_main_proc:
                    pbar.update(batch.batch_size)

            with Timer((name, 'GPU compute'), self.log) as timer:
                if hasattr(self.model, 'module'):
                    self.model_noddp.load_state_dict(
                        self.model.module.state_dict())
                else:
                    self.model_noddp.load_state_dict(self.model.state_dict())
                result = test.batchwise_test(
                    self.model_noddp, len(loader), devit, cb)

                timer.stop()
                if self.is_main_proc:
                    pbar.close()
                    del pbar

            if dist.is_initialized():
                output_0 = torch.tensor([result[0]]).to(self.main_device)
                output_1 = torch.tensor([result[1]]).to(self.main_device)
                _ = dist.all_reduce(output_0)
                _ = dist.all_reduce(output_1)
                result = (output_0.item(), output_1.item())
            results[name] = result[0] / result[1]

        return results
