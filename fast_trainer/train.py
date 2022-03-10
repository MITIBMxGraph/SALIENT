from io import UnsupportedOperation
from typing import Optional
import torch
import torch.nn.functional as F

from .samplers import PreparedBatch
from .transferers import DeviceIterator
from .concepts import TrainCore, TrainCallback
import torch.distributed

from .utils import runtime_stats_cuda, is_performance_stats_enabled


def barebones_train_core(model: torch.nn.Module, batch: PreparedBatch):
    out = model(batch.x, batch.adjs)
    loss = F.nll_loss(out, batch.y)
    loss.backward()
    return loss


def make_eval_and_loss(module, train_core):
    def eval_and_loss(*args, **_):
        return train_core(module, PreparedBatch(*args))

    return eval_and_loss


def data_parallel_train(model: torch.nn.Module,
                        train_core: TrainCore,
                        devit: DeviceIterator,
                        optimizer: torch.optim.Optimizer, lr_scheduler,
                        cb: Optional[TrainCallback] = None,
                        dataset=None,
                        devices=None) -> None:
    model.train()
    while True:
        optimizer.zero_grad()

        # Replicate the model (send weights) to devices
        #
        # TODO: This might not be non-blocking. If so, this is a
        #       PyTorch issue!
        #
        # NOTE: This creates "replica modules" whose gradients are
        #       automatically reduced during the computation of
        #       the backward pass.
        replicas = torch.nn.parallel.replicate(model, devit.devices)
        inputs = next(devit, [])

        if len(inputs) == 0:
            break

        replicas = replicas[:len(inputs)]
        devices = devit.devices[:len(inputs)]

        funcs = [make_eval_and_loss(replica, train_core)
                 for replica in replicas]

        # NOTE: devices can be inferred from inputs, but providing
        # them is faster
        results = torch.nn.parallel.parallel_apply(
            funcs, inputs, devices=devices)

        optimizer.step()

        if lr_scheduler is not None:
            for batch_res in results:
                lr_scheduler.step(batch_res)

        if cb is not None:
            cb(inputs, results)

        # skip replicating next iter, if we have no more data
        if len(inputs) < len(devit.devices):
            break

        del inputs
        del results
        del funcs


def serial_train_ns(model: torch.nn.Module,
                    train_core: TrainCore,
                    devit: DeviceIterator,
                    optimizer: torch.optim.Optimizer,
                    lr_scheduler,
                    cb: Optional[TrainCallback] = None,
                    dataset=None,
                    devices=None) -> None:
    ''' Serial training code that uses PyG's NeighborSampler '''
    model.train()

    runtime_stats_cuda.start_epoch()

    if devices is not None:
        assert len(devices) == 1
        device = devices[0]

    runtime_stats_cuda.start_region("total")
    runtime_stats_cuda.start_region(
        "sampling", runtime_stats_cuda.get_last_event())
    iterator = iter(devit)
    runtime_stats_cuda.end_region("sampling")
    runtime_stats_cuda.end_region("total", runtime_stats_cuda.get_last_event())

    while True:
        runtime_stats_cuda.start_region(
            "total", runtime_stats_cuda.get_last_event())
        runtime_stats_cuda.start_region(
            "sampling", runtime_stats_cuda.get_last_event())
        inputs = next(iterator, [])
        if len(inputs) == 0:
            runtime_stats_cuda.end_region("sampling")
            runtime_stats_cuda.end_region(
                "total", runtime_stats_cuda.get_last_event())
            break

        batch_size, n_id, adjs = inputs
        xs = torch.empty(len(n_id), dataset.x.shape[1], dtype=dataset.x.dtype,
                         layout=dataset.x.layout, pin_memory=True)
        torch.index_select(dataset.x, 0, n_id, out=xs)
        ys = torch.empty(batch_size, dtype=dataset.y.dtype,
                         layout=dataset.y.layout, pin_memory=True)
        torch.index_select(dataset.y, 0, n_id[:batch_size], out=ys)

        runtime_stats_cuda.end_region("sampling")

        runtime_stats_cuda.start_region(
            "data_transfer", runtime_stats_cuda.get_last_event())
        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        adjs = [adj.to(device, non_blocking=True) for adj in adjs]
        runtime_stats_cuda.end_region("data_transfer")

        runtime_stats_cuda.start_region(
            "train", runtime_stats_cuda.get_last_event())
        optimizer.zero_grad()
        out = model(xs, adjs)
        loss = F.nll_loss(out, ys)
        loss.backward()
        result = loss
        optimizer.step()

        if lr_scheduler is not None:
            world_size = 1.0
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(result)
                world_size = 1.0*torch.distributed.get_world_size()
            lr_scheduler.step(result / world_size)

        if cb is not None:
            cb([inputs], [result])
        runtime_stats_cuda.end_region("train")
        runtime_stats_cuda.end_region(
            "total", runtime_stats_cuda.get_last_event())
    runtime_stats_cuda.end_epoch()
    runtime_stats_cuda.report_stats(
        {'total': 'Total', 'data_transfer': 'Data Transfer', 'sampling': 'Sampling + Slicing', 'train': 'Train'})


def serial_train_with_timing(model: torch.nn.Module,
                             train_core: TrainCore,
                             devit: DeviceIterator,
                             optimizer: torch.optim.Optimizer,
                             lr_scheduler,
                             cb: Optional[TrainCallback] = None,
                             dataset=None,
                             devices=None) -> None:
    '''
    Serial train code that uses SALIENT's FastSampler with basic instrumentation to time different operations.
    This function is only designed for single GPU non-distributed setting.
    '''
    if dataset is not None and devices is not None:
        serial_train_ns(model, train_core, devit, optimizer, lr_scheduler,
                        cb=cb, dataset=dataset, devices=devices)
        return

    model.train()

    runtime_stats_cuda.start_region("total")
    runtime_stats_cuda.end_region("total")

    runtime_stats_cuda.start_epoch()
    while True:
        runtime_stats_cuda.start_region(
            "total", runtime_stats_cuda.get_last_event())
        runtime_stats_cuda.start_region(
            "load_batch", runtime_stats_cuda.get_last_event())
        try:
            inp, = next(devit)
            # The sampling region is opened, but not closed by next method of devit.
            runtime_stats_cuda.end_region("sampling")
        except StopIteration:
            # The sampling region is opened, but not closed by next method of devit.
            runtime_stats_cuda.end_region("sampling")
            runtime_stats_cuda.end_region(
                "load_batch", runtime_stats_cuda.get_last_event())
            runtime_stats_cuda.end_region(
                "total", runtime_stats_cuda.get_last_event())
            break
        runtime_stats_cuda.end_region(
            "load_batch", runtime_stats_cuda.get_last_event())

        runtime_stats_cuda.start_region(
            "train", runtime_stats_cuda.get_last_event())
        optimizer.zero_grad()
        result = train_core(model, inp)
        optimizer.step()

        # Use of the LR in the loop here may cause a performance penalty.
        if lr_scheduler is not None:
            world_size = 1.0
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(result)
                world_size = 1.0*torch.distributed.get_world_size()
            lr_scheduler.step(result / world_size)

        if cb is not None:
            cb([inp], [result])
        runtime_stats_cuda.end_region("train")
        runtime_stats_cuda.end_region("total")
    runtime_stats_cuda.end_epoch()
    runtime_stats_cuda.report_stats(
        {'total': 'Total', 'data_transfer': 'Data Transfer', 'sampling': 'Sampling + Slicing', 'train': 'Train'})


def serial_train(model: torch.nn.Module,
                 train_core: TrainCore,
                 devit: DeviceIterator,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler,
                 cb: Optional[TrainCallback] = None,
                 dataset=None,
                 devices=None) -> None:
    ''' Serial train code that uses SALIENT's FastSampler '''
    if dataset is not None and devices is not None:
        serial_train_ns(model, train_core, devit, optimizer, lr_scheduler,
                        cb=cb, dataset=dataset, devices=devices)
        return

    if is_performance_stats_enabled():
        serial_train_with_timing(model, train_core, devit, optimizer, lr_scheduler,
                                 cb=cb, dataset=dataset, devices=devices)
        return

    model.train()

    iterator = iter(devit)

    while True:
        try:
            inp, = next(iterator)
        except StopIteration:
            break
        optimizer.zero_grad()
        result = train_core(model, inp)
        optimizer.step()
        if lr_scheduler is not None:
            world_size = 1.0
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(result)
                world_size = 1.0*torch.distributed.get_world_size()
            lr_scheduler.step(result.cpu()/world_size)
        if cb is not None:
            cb([inp], [result])
