"""
Exponential moving average applied to model paras

Based on SWA in torchcontrib
"""

from torchcontrib.optim import SWA
import torch
from functools import wraps

class EMA(SWA):
    def __init__(self, optimizer, ema_start=1, ema_freq=1, ema_lr=None, decay=0.99, zero_debias=False):
        super().__init__(optimizer, swa_start=ema_start, swa_freq=ema_freq, swa_lr=ema_lr)
        self.decay = decay
        self.zero_debias = zero_debias

    def swap_ema_sgd(self):
        self.swap_swa_sgd()

    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.
        This can be useful when fine tuning a pre-trained network as frozen
        layers can be made trainable and added to the :class:`Optimizer` as
        training progresses.
        Args:
            param_group (dict): Specifies what Tensors should be optimized along
            with group specific optimization options.
        """

        param_group['step_counter'] = 0
        self.optimizer.add_param_group(param_group)

    def update_swa_group(self, group):
        r"""Updates the SWA running averages for the given parameter group.
        Arguments:
            param_group (dict): Specifies for what parameter group SWA running
                averages should be updated
        Examples:
            >>> # automatic mode
            >>> base_opt = torch.optim.SGD([{'params': [x]},
            >>>             {'params': [y], 'lr': 1e-3}], lr=1e-2, momentum=0.9)
            >>> opt = torchcontrib.optim.SWA(base_opt)
            >>> for i in range(100):
            >>>     opt.zero_grad()
            >>>     loss_fn(model(input), target).backward()
            >>>     opt.step()
            >>>     if i > 10 and i % 5 == 0:
            >>>         # Update SWA for the second parameter group
            >>>         opt.update_swa_group(opt.param_groups[1])
            >>> opt.swap_swa_sgd()
        """
        for p in group['params']:
            param_state = self.state[p]
            if 'swa_buffer' not in param_state:
                if self.zero_debias:
                    param_state['swa_buffer'] = torch.empty_like(p.data)
                    param_state['swa_buffer'].copy_(p.data)
                else:
                    param_state['swa_buffer'] = torch.zeros_like(p.data)
                
            buf = param_state['swa_buffer']
            # virtual_decay = 1 / float(group["n_avg"] + 1)
            diff = (p.data - buf) * (1 - self.decay)
            buf.add_(diff)
        # group["n_avg"] += 1

    def state_dict(self):
        state = super().state_dict()
        if isinstance(self.decay, torch.Tensor):
            decay = self.decay.cpu().item()
        else:
            decay = self.decay
        state['ema_decay'] = decay
        return state

    def load_state_dict(self, state_dict):
        decay = state_dict.pop('ema_decay')
        if isinstance(self.decay, torch.Tensor):
            self.decay.fill_(decay)
        else:
            self.decay = decay

        super().load_state_dict(state_dict)
        
def ema_autoswap(ema: EMA):
    """
    Decorator that helps automatically switch between 
    ema parameters and original parameters during validation
    """

    assert isinstance(ema, EMA)
    def ema_dec(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ema.swap_ema_sgd()
            rtn = func(*args, **kwargs)
            ema.swap_ema_sgd()
            return rtn
        return wrapper
    return ema_dec
