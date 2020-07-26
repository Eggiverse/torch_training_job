import inspect
import math
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict

import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

from .data.fakeloader import FakeLoader

__all__ = ['MultiStepTs', 'LinearIncreaseTs', 'ExpIncreaseTs',
           'get_ts_scheduler', 'SchedStepMode', 'scheduler_step',
           'LR_SCHED_DICT', 'AbstractScheduler', 'FakeLoaderScheduler']

class AbstractScheduler(ABC):
    @abstractmethod
    def step(self):
        ...

    @abstractmethod
    def state_dict(self):
        ...

    @abstractmethod
    def load_state_dict(self, state_dict):
        ...

    @classmethod
    def __subclasshook__(cls, C):
        if cls is AbstractScheduler:
            if any("step" in B.__dict__ for B in C.__mro__):
                return True
        return NotImplemented

class SchedStepMode(Enum):
    epoch = 1
    iteration = 2

    @classmethod
    def get_mode(cls, mode):
        return cls.__members__[mode]

## Begin TS
class _TensorScheduler(AbstractScheduler):

    def __init__(self, para: torch.Tensor, last_epoch=-1):
        self.para = para
        self.last_epoch = last_epoch

        self.step()

    def get_para(self):
        raise NotImplementedError

    def step(self):
        self.last_epoch += 1
        self.para.fill_(self.get_para())

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict):
        local_state = self.__dict__
        for key, val in state_dict.items():
            if key in state_dict:
                if not isinstance(val, torch.Tensor):
                    local_state[key] = val
                else:
                    local_val = local_state[key]
                    try:
                        with torch.no_grad():
                            local_val.copy_(val)
                    except Exception as ex:
                        raise ValueError('While copying the parameter named "{}", '
                                         'whose dimensions in the model are {} and '
                                         'whose dimensions in the checkpoint are {}, '
                                         'an exception occured : {}.'
                                         .format(key, local_val.size(), val.size(), ex.args))


class MultiStepTs(_TensorScheduler):

    def __init__(self, para:torch.Tensor, milestones:Dict[int, float], last_epoch=-1):
        self.milestones = milestones

        if last_epoch != -1:
            warnings.warn(f"{self.__class__} does not get correct value based on last epoch!")

        super().__init__(para, last_epoch=last_epoch)

    def get_para(self):
        if self.last_epoch in self.milestones.keys():
            return self.milestones[self.last_epoch]
        return self.para.item()
        # key = bisect_right(list(self.milestones.keys()), self.last_epoch)
        # return list(self.milestones.values())[key]

class _IncreaseTs(_TensorScheduler):

    def __init__(self, para: torch.Tensor, k=None, start_epoch=0, end_epoch=None, end_val=None, last_epoch=-1):
        self.init_para = para.item()
        if k is None:
            self.k = self.get_k(self.init_para, end_val, start_epoch, end_epoch)
        else:
            self.k = k
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

        super().__init__(para, last_epoch=last_epoch)

    @staticmethod
    def get_k(init_val, end_val, start_epoch, end_epoch):
        raise NotImplementedError

    def calc_para(self):
        raise NotImplementedError

    def get_para(self):
        if self.end_epoch is not None and self.last_epoch > self.end_epoch or self.last_epoch < self.start_epoch:
            return self.para.item()

        return self.calc_para()

class LinearIncreaseTs(_IncreaseTs):
    @staticmethod
    def get_k(init_val, end_val, start_epoch, end_epoch):
        return (end_val - init_val) / (end_epoch - start_epoch)

    def calc_para(self):
        return self.k * (self.last_epoch - self.start_epoch) + self.init_para#, self.init_para)

class ExpIncreaseTs(_IncreaseTs):
    # @staticmethod
    # def get_k(init_val, end_val, start_epoch, end_epoch):
    #     return (end_val - init_val) / (end_epoch - start_epoch)

    def calc_para(self):
        t = (self.last_epoch - self.start_epoch)
        return max(math.exp(self.k*t) - 1, 0) * self.init_para

TSSCHED_DICT = {
    "multistep": MultiStepTs,
    "linear": LinearIncreaseTs,
    "exp": ExpIncreaseTs
}

def get_ts_scheduler(para: torch.Tensor, sched_config, last_epoch=-1) -> _TensorScheduler:
    sched_type = TSSCHED_DICT[sched_config.pop("type", "multistep")]
    config = sched_config.get("config", sched_config)
    return sched_type(para, **config, last_epoch=last_epoch)
### End Tensor

### Begin LR

class RangeLambdaLR(LambdaLR):

    def range_lambda(self, lmbda, epoch, base_lr, current_lr):
        if (ratio:=lmbda(epoch)) < 0:
            new_lr = current_lr
            self.base_lrs = list(map(lambda group: group['lr'], self.optimizer.param_groups))
        else:
            new_lr = base_lr * ratio
        return new_lr

    def get_lr(self):
        return [self.range_lambda(lmbda, self.last_epoch, base_lr, group['lr'])
                for lmbda, base_lr, group in zip(self.lr_lambdas, self.base_lrs, self.optimizer.param_groups)]

def LinearDecreaseLR(optimizer, start_epoch, end_epoch, endlr=0, last_epoch=-1):
    def decrease_function(epoch):
        if epoch < start_epoch:
            return -1
        if epoch <= end_epoch:
            return 1 - (epoch - start_epoch) / (end_epoch - start_epoch) * (1-endlr)

        return -1#endlr
    return RangeLambdaLR(optimizer, decrease_function, last_epoch=last_epoch)

class MyMultiStepLR(_LRScheduler):

    def __init__(self, optimizer, milestones:Dict[int, float], last_epoch=-1):
        self.milestones = milestones
        # self.gamma = gamma

        if last_epoch != -1:
            warnings.warn(f"{self.__class__} does not get correct value based on last epoch!")
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch not in self.milestones.keys():
            return [group['lr'] for group in self.optimizer.param_groups]
        # return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
        #         for group in self.optimizer.param_groups]
        return [self.milestones[self.last_epoch]] * len(self.optimizer.param_groups)


LR_SCHED_DICT = dict(inspect.getmembers(optim.lr_scheduler, lambda x: inspect.isclass(x) and
                                        issubclass(x, torch.optim.lr_scheduler._LRScheduler) and
                                        x is not torch.optim.lr_scheduler._LRScheduler))

LR_SCHED_DICT.update({
    "Plateau": optim.lr_scheduler.ReduceLROnPlateau,
    "MyMultiStepLR": MyMultiStepLR,
    "Linear": LinearDecreaseLR
})

## End LR

class FakeLoaderScheduler(AbstractScheduler):

    def __init__(self, fakeloader: FakeLoader, open_epoch:int, last_epoch=0):
        self.loader = fakeloader
        self.last_epoch = last_epoch
        self.open_epoch = open_epoch

    def get_para(self):
        return self.loader

    def step(self):
        self.last_epoch += 1
        if self.last_epoch == self.open_epoch:
            self.loader.init_gate()

    def state_dict(self):
        return {key: val for key, val in self.__dict__.items() if not isinstance(val, FakeLoader)}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        if self.last_epoch >= self.open_epoch:
            self.loader.init_gate()


def scheduler_step(sched: AbstractScheduler, metric=None):
    if sched is None:
        return
    if isinstance(sched, optim.lr_scheduler.ReduceLROnPlateau):
        sched.step(metric)
    else:
        sched.step()
