import torch


class DistributedShuffler:
    initial_idx: torch.Tensor
    current_idx: torch.Tensor
    world_size: int
    initial_seed: int
    generator: torch.Generator
    epoch: int

    def __init__(self, idx, world_size, initial_seed=2147483647):
        self.initial_idx = idx
        self.world_size = world_size
        self.initial_seed = initial_seed
        self.generator = torch.Generator(device='cpu')
        self.set_epoch(0)

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.generator.manual_seed(self.initial_seed + epoch)
        self.current_idx = self.initial_idx[
            torch.randperm(self.initial_idx.numel(),
                           generator=self.generator,
                           device=self.initial_idx.device)]

    def get_indices(self, rank):
        n = self.current_idx.numel()
        start = (n * rank) // self.world_size
        stop = (n * (rank + 1)) // self.world_size
        return self.current_idx[start: stop]
