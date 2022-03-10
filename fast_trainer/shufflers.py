import torch


class Shuffler:
    initial_idx: torch.Tensor
    world_size: int
    initial_seed: int
    generator: torch.Generator
    epoch: int

    DEFAULT_INITIAL_SEED = 2147483647

    def __init__(self, idx, initial_seed=DEFAULT_INITIAL_SEED):
        assert idx.dim() == 1
        self.initial_idx = idx
        self.initial_seed = initial_seed
        self.generator = torch.Generator(device='cpu')
        self.set_epoch(0)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def get_idx(self):
        self.generator.manual_seed(self.initial_seed + self.epoch)
        return self.initial_idx[torch.randperm(self.initial_idx.numel(),
                                               generator=self.generator,
                                               device=self.initial_idx.device)]


class DistributedShuffler(Shuffler):
    world_size: int

    def __init__(self, idx, world_size,
                 initial_seed=Shuffler.DEFAULT_INITIAL_SEED):
        super().__init__(idx, initial_seed)
        self.world_size = world_size

    def get_idx(self, rank):
        shuffled_idx = super().get_idx()
        n = shuffled_idx.numel()
        start = (n * rank) // self.world_size
        stop = (n * (rank + 1)) // self.world_size
        return shuffled_idx[start: stop]
