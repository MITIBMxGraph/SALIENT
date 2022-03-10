from . import BaseDriver
from fast_trainer.shufflers import Shuffler


class SingleProcDriver(BaseDriver):
    def __init__(self, args, devices, dataset, model_type):
        super().__init__(args, devices, dataset, model_type)

        self.train_shuffler = Shuffler(self.dataset.split_idx['train'])

    def get_idx(self, epoch: int):
        self.train_shuffler.set_epoch(epoch)
        return self.train_shuffler.get_idx()

    def get_idx_test(self, name):
        return self.dataset.split_idx[name]

    @property
    def is_main_proc(self):
        return True
