import os
import time
from typing import NamedTuple
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from . import BaseDriver
from fast_trainer.shufflers import DistributedShuffler


def set_master(addr: str, port=1884):
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = str(port)


class DDPConfig(NamedTuple):
    node_num: int
    num_devices_per_node: int
    total_num_nodes: int

    @property
    def world_size(self):
        return self.total_num_nodes * self.num_devices_per_node


def get_ddp_config(job_dir: str, total_num_nodes: int,
                   num_devices_per_node: int):
    assert total_num_nodes > 0
    assert num_devices_per_node > 0

    my_node_name = str(os.environ['SLURMD_NODENAME'])
    print(f'{my_node_name} starting', flush=True)

    print("DDP dir is " + str(job_dir), flush=True)
    while True:
        node_list = os.listdir(job_dir)
        print("Node list is " + str(node_list), flush=True)
        print("Length is " + str(len(node_list)) + " and waiting for "
              + str(total_num_nodes), flush=True)
        if len(node_list) == total_num_nodes:
            break
        time.sleep(1)

    node_list = sorted(node_list)

    for i in range(len(node_list)):
        if my_node_name == node_list[i]:
            node_num = i
            break
    else:
        raise ValueError(f'Unable to find {my_node_name} in {node_list}')

    set_master(node_list[0])

    return DDPConfig(node_num=node_num,
                     num_devices_per_node=num_devices_per_node,
                     total_num_nodes=len(node_list))


class DDPDriver(BaseDriver):
    global_rank: int
    ddp_cfg: DDPConfig

    def __init__(self, args, device, dataset, model_type, ddp_cfg: DDPConfig):
        assert args.train_type == 'serial'

        self.ddp_cfg = ddp_cfg
        self.global_rank = (
            ddp_cfg.node_num * ddp_cfg.num_devices_per_node + device.index)
        dist.init_process_group(
            'nccl', rank=self.global_rank, world_size=ddp_cfg.world_size)

        self.orig_model = None
        super().__init__(args, [device], dataset, model_type)

        self.train_shuffler = DistributedShuffler(
            self.dataset.split_idx['train'], ddp_cfg.world_size)
        self.test_shuffler = DistributedShuffler(
            self.dataset.split_idx['test'], ddp_cfg.world_size)
        self.valid_shuffler = DistributedShuffler(
            self.dataset.split_idx['valid'], ddp_cfg.world_size)
        self.reset()

    def __del__(self):
        dist.destroy_process_group()

    def _reset_model(self):
        if self.orig_model is None:
            self.orig_model = self.model
        self.orig_model.reset_parameters()
        self.model = DistributedDataParallel(
            self.orig_model, device_ids=[self.main_device])  # , find_unused_parameters=True)

    def get_idx_test(self, name):
        if name == 'test':
            return self.test_shuffler.get_idx(self.global_rank)
        elif name == 'valid':
            return self.valid_shuffler.get_idx(self.global_rank)
        else:
            raise ValueError('invalid test dataset name')

    def get_idx(self, epoch: int):
        self.train_shuffler.set_epoch(10000*self.TRIAL_NUM + epoch)
        return self.train_shuffler.get_idx(self.global_rank)

    def get_sampler(self, _seed):
        return torch.utils.data.distributed.DistributedSampler(
            self.dataset.split_idx['train'],
            num_replicas=self.ddp_cfg.world_size,
            rank=self.global_rank, seed=_seed)

    @property
    def my_name(self):
        return f'{super().my_name}_{self.ddp_cfg.node_num}_{self.main_device.index}'

    @property
    def is_main_proc(self):
        return self.global_rank == 0
