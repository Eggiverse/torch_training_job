"""
Optimizer utils
"""

import inspect
from typing import Any, Dict, Iterable, List, Union

import torch.nn as nn
import torch.optim as optim

__all__ = [
    'get_opt',
    'set_momentum'
]


OPTIM_DICT = dict(inspect.getmembers(optim, lambda x: inspect.isclass(x) and
                                        issubclass(x, optim.Optimizer) and
                                        x is not optim.Optimizer))

def get_opt(config: Dict[str, Any], paras: Iterable[nn.Parameter])-> optim.Optimizer:
    opt_type = config.pop('opt_type', 'Adam')

    return OPTIM_DICT[opt_type](paras, **config["opt"])

def set_momentum(opt: optim.SGD,
                 mom: Union[float, List[float]]):
    """Set default momentum of SGD
    """
    if isinstance(mom, list):
        for group, m in zip(opt.param_groups, mom):
            group['momentum'] = m
    else:
        for group in opt.param_groups:
            group['momentum'] = mom
