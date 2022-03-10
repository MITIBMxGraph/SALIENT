import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import shutil
from pathlib import Path
import os
import collections
from itertools import repeat
from typing import List, Dict, Any

from .dataset import FastDataset
from .drivers import *
from .drivers.ddp import DDPConfig, get_ddp_config, set_master
from .models import SAGE, GAT, GIN, SAGEResInception
from .models import SAGEClassic, JKNet, GCN, ARMA
from .parser import make_parser

from fast_trainer.utils import Timer
from fast_trainer.utils import setup_runtime_stats, report_runtime_stats
from fast_trainer.utils import enable_runtime_stats, disable_runtime_stats


def get_dataset(dataset_name, root):
    assert dataset_name in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']
    return FastDataset.from_path(root, dataset_name)


def consume_prefix_in_state_dict_if_present(
    state_dict: Dict[str, Any], prefix: str
) -> None:
    r"""Strip the prefix in state_dict in place, if any.

    ..note::
        Given a `state_dict` from a DP/DDP model, a local model can
        load it by applying
        `consume_prefix_in_state_dict_if_present(state_dict,
        "module.")` before calling
        :meth:`torch.nn.Module.load_state_dict`.

    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.

    """

    #state_dict = _state_dict.copy()
    keys = sorted(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix):]
            state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata if any.
    if "_metadata" in state_dict:
        metadata = state_dict["_metadata"]
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix):]
            metadata[newkey] = metadata.pop(key)
    return state_dict


def get_model_type(model_name):
    assert model_name.lower() in ['sage', 'gat', 'gin', 'sageresinception',
                                  'sageclassic', 'jknet', 'gcn', 'arma']

    if model_name.lower() == 'sage':
        return SAGE                                                   # works
    if model_name.lower() == 'gat':
        return GAT                                                    # works
    if model_name.lower() == 'gin':
        return GIN              # works. does not support layerwise inference
    if model_name.lower() == 'sageresinception':
        return SAGEResInception # works. does not support layerwise inference
    if model_name.lower() == 'sageclassic':
        return SAGEClassic                                # not used in paper
    if model_name.lower() == 'jknet':
        return JKNet                                      # not used in paper
    if model_name.lower() == 'gcn':
        return GCN                                        # not used in paper
    if model_name.lower() == 'arma':
        return ARMA                                                  # broken


def get_job_dir(args):
    return Path(args.output_root).joinpath(args.job_name)


def run_driver(args, drv: BaseDriver):
    trial_results = []
    setup_runtime_stats(args)
    for TRIAL in range(0, args.trials):
        drv.reset()
        # drv.model.module.reset_parameters()
        delta = min(args.test_epoch_frequency, args.epochs)
        do_eval = args.epochs >= args.test_epoch_frequency
        best_acc = 0
        best_acc_test = 0
        job_dir = get_job_dir(args)
        best_epoch = None

        if args.do_test_run:
            for i in range(0, len(args.do_test_run_filename)):
                if isinstance(drv, DDPDriver):
                    drv.model.module.load_state_dict(
                        consume_prefix_in_state_dict_if_present(
                            torch.load(args.do_test_run_filename[i]),
                            'module.'))
                else:
                    drv.model.load_state_dict(
                        consume_prefix_in_state_dict_if_present(
                            torch.load(args.do_test_run_filename[i]),
                            'module.'))
                if isinstance(drv, DDPDriver):
                    dist.barrier()
                if drv.is_main_proc:
                    print("\nPerforming inference on trained model at " +
                          args.do_test_run_filename[i])
                acc = drv.test(('test',))['test']
                if isinstance(drv, DDPDriver):
                    dist.barrier()
                if drv.is_main_proc:
                    print("Final test accuracy is: " + str(acc))
                trial_results.append((acc, args.do_test_run_filename[i]))
                drv.flush_logs()
            break

        if drv.is_main_proc:
            print()
            print("+" + "-"*40 + "+")
            print("+" + " "*16 + "TRIAL "+"{:2d}".format(TRIAL) + " "*16 + "+")
            print("+" + "-"*40 + "+")

        if drv.is_main_proc:
            print("Performing training")
        for epoch in range(0, args.epochs, delta):
            if isinstance(drv, DDPDriver):
                dist.barrier()
            enable_runtime_stats()
            drv.train(range(epoch, epoch + delta))
            disable_runtime_stats()
            if do_eval:
                if isinstance(drv, DDPDriver):
                    dist.barrier()
                acc_type = 'valid'
                acc = drv.test((acc_type,))[acc_type]
                if drv.is_main_proc:
                    drv.log((acc_type, 'Accurracy', acc))
                if acc > best_acc:
                    best_acc = acc
                    this_epoch = epoch + delta - 1
                    best_epoch = this_epoch
                    if drv.is_main_proc:
                        torch.save(
                            drv.model.state_dict(),
                            job_dir.joinpath(f'model_{TRIAL}_{this_epoch}.pt'))
                        with job_dir.joinpath('metadata.txt').open('a') as f:
                            f.write(','.join(map(str, (this_epoch, acc))))
                            f.write('\n')
                if drv.is_main_proc:
                    print("Best validation accuracy so far: " + str(best_acc))
                if isinstance(drv, DDPDriver):
                    dist.barrier()
            drv.flush_logs()

        report_runtime_stats(drv.log)

        if drv.is_main_proc:
            print("\nPerforming inference on trained model at " + 
                  str(job_dir.joinpath(f'model_{TRIAL}_{best_epoch}.pt')))
        drv.model.load_state_dict(torch.load(
            job_dir.joinpath(f'model_{TRIAL}_{best_epoch}.pt')))
        acc_type = 'valid'
        acc = drv.test((acc_type,))[acc_type]
        if drv.is_main_proc:
            drv.log((acc_type, 'Accuracy', acc))
        if isinstance(drv, DDPDriver):
            dist.barrier()
        final_valid_acc = acc
        acc_type = 'test'
        acc_test = drv.test((acc_type,))[acc_type]
        if isinstance(drv, DDPDriver):
            dist.barrier()
        if drv.is_main_proc:
            drv.log((acc_type, 'Accuracy', acc_test))
        final_test_acc = acc_test
        trial_results.append((final_valid_acc, final_test_acc))
        if drv.is_main_proc:
            print("\nFinal validation,test accuracy is: " +
                  str(final_valid_acc) + "," + str(final_test_acc) +
                  " on trial " + str(TRIAL))

    if drv.is_main_proc:
        print("")
        drv.log(('End results for all trials', str(trial_results)))
    drv.flush_logs()


def ddp_main(rank, args, dataset, model_type, ddp_cfg: DDPConfig):
    device = torch.device(type='cuda', index=rank)
    drv = DDPDriver(args, device, dataset, model_type, ddp_cfg)
    run_driver(args, drv)


def add_labels(feat, labels, idx, n_classes):
    onehot = torch.zeros([feat.shape[0], n_classes])  # , device=device)
    onehot[idx, labels[idx]] = 1
    return torch.cat([feat, onehot], dim=-1)


if __name__ == '__main__':
    #assert torch.cuda.is_available()

    args = make_parser().parse_args()
    job_dir = get_job_dir(args)

    assert args.num_layers == len(args.train_fanouts)
    assert args.num_layers == len(args.batchwise_test_fanouts)

    do_ddp = args.total_num_nodes > 1 or args.one_node_ddp

    if not do_ddp:
        if job_dir.exists():
            assert job_dir.is_dir()
            if args.overwrite_job_dir:
                shutil.rmtree(job_dir)
            else:
                raise ValueError(
                    f'job_dir {job_dir} exists. Use a different job name ' +
                    'or set --overwrite_job_dir')
        job_dir.mkdir(parents=True)

    num_devices_per_node = min(args.max_num_devices_per_node,
                               torch.cuda.device_count())

    print(f'Using {num_devices_per_node} devices per node')

    if args.train_sampler == 'NeighborSampler' and args.train_type != 'serial':
        raise ValueError(
            'The dp (data parallel) train_type is not supported by this ' +
            'driver for the train_sampler NeighborSampler.')

    with Timer('Loading dataset'):
        dataset = get_dataset(args.dataset_name, args.dataset_root)

    model_type = get_model_type(args.model_name)

    SLURM_name = str(os.environ['SLURMD_NODENAME'])

    # Write the args to the job dir for reproducibility
    with get_job_dir(args).joinpath(SLURM_name + "args.txt").open('w') as f:
        f.write(repr(args))

    # HACK: Pass the log_file BaseDriver
    # I just got lazy and didn't want to add another constructor argument.
    # This is fine because args is saved right above.
    args.log_file = job_dir.joinpath(SLURM_name+"logs.txt")

    if do_ddp:

        if args.total_num_nodes == 1:
            assert args.one_node_ddp
            print("Fall into 1 node ddp")
            set_master('localhost')
            ddp_cfg = DDPConfig(0, num_devices_per_node, 1)
        else:
            ddp_cfg = get_ddp_config(args.ddp_dir, args.total_num_nodes,
                                     num_devices_per_node)

        print(f'Using DDP with {ddp_cfg.total_num_nodes} nodes')
        with Timer('dataset.share_memory_()'):
            dataset.share_memory_()

        # Note: On smaller GPU instances with limited memory (e.g. 128
        # GB), the NeighborSampler fails when spawning
        # subprocesses. This issue doesn't impact the
        # FastSampler/SALIENT, and it seems to be avoided by not
        # spawning a separate subprocess for NeighborSampler.
        # Unfortunately: Attempted workarounds cause the process to
        # not halt at the end of training. For now, PyG distributed
        # experiments require machines with more main memory --- about
        # 256GB for ogbn-papers100M.
        mp.spawn(ddp_main, args=(args, dataset, model_type, ddp_cfg),
                 nprocs=num_devices_per_node, join=True)

    else:

        devices = [torch.device(type='cuda', index=i)
                   for i in range(num_devices_per_node)]
        print(f'Using {args.train_type} training')
        drv = SingleProcDriver(args, devices, dataset, model_type)
        run_driver(args, drv)
