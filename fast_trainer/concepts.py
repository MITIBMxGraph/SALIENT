from typing import Any, Optional, Callable, List
import torch

from .samplers import PreparedBatch
from .transferers import DeviceIterator

TrainCore = Callable[[torch.nn.Module, PreparedBatch], Any]
TrainCallback = Callable[[List[PreparedBatch], List[Any]], None]
TrainImpl = Callable[[torch.nn.Module, TrainCore, DeviceIterator,
                      torch.optim.Optimizer, Optional[TrainCallback]], None]
TestCallback = Callable[[PreparedBatch], None]  # should not return anything
