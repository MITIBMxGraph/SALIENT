import torch

from .transferers import DeviceIterator
from .concepts import TestCallback


@torch.no_grad()
def batchwise_test(model: torch.nn.Module,
                   num_batches: int,
                   devit: DeviceIterator,
                   cb: TestCallback = None):
    model.eval()

    device, = devit.devices

    results = torch.empty(num_batches, dtype=torch.long, pin_memory=True)
    total = 0

    for i, inputs in enumerate(devit):
        inp, = inputs

        out = model(inp.x, inp.adjs)
        out = out.argmax(dim=-1, keepdim=True).reshape(-1)
        correct = (out == inp.y).sum()
        results[i].copy_(correct, non_blocking=True)
        total += inp.batch_size

        if cb is not None:
            cb(inp)

    torch.cuda.current_stream(device).synchronize()

    return results.sum().item(), total
