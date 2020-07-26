import warnings

import torch
import torch.nn.functional as F

from .metrics import average_helper


class ModernLoss:
    def __init__(self, loss, tag):
        self.tags = [tag]
        self.loss = loss

    def from_model(self, model_out, batch_sample):
        target = batch_sample[1]
        return MultiLoss(
            main_loss_tag = self.tags[0],
            **{self.tags[0]: self.loss(model_out, target)})

def _binary_entropy(vec):
    p = torch.sigmoid(vec)
    logp = F.logsigmoid(vec)
    log1p = F.logsigmoid(-vec)
    return - (p * logp + (1-p) * log1p).mean()

def entropy(vec: torch.Tensor) -> torch.Tensor:
    """
    Entropy of output of a typical classification model

    vec is the output before softmax
    """
    if vec.ndim == 1 or vec.size(1) == 1:
        return _binary_entropy(vec)

    p = F.softmax(vec, dim=1)
    logp = F.log_softmax(vec, dim=1)
    return - (p * logp).sum(dim=1).mean(dim=0)

class MultiLoss:
    """
    Helper for muiltiple loss from semi-supervised/multi-task learning

    It mimics the behaviour of a single loss when calc grad

    """
    data_size: int
    def __init__(self, main_loss_tag="mix_loss", sched_metric_tag=None, **kwargs):
        self.main_loss = None
        if main_loss_tag in kwargs.keys():
            self.main_loss = kwargs[main_loss_tag]

        self.loss_dict = kwargs

        self.sched_metric = None
        if sched_metric_tag is not None and sched_metric_tag in kwargs.keys():
            self.sched_metric = kwargs[sched_metric_tag]
        
        
    def values(self):
        return self.loss_dict.values()

    def backward(self, **kwargs):
        self.main_loss.backward(**kwargs)

    @classmethod
    def getAccumulater(cls, metric_tags):
        ml = cls(**{
            key: 0. for key in metric_tags
        })
        ml.data_size = 0
        ml.report = None
        return ml

    def accum(self, other, batchsize=1):
        for key, val in other.loss_dict.items():
            self.loss_dict[key] += val.item() * batchsize
    
        self.data_size += batchsize

    def get_average(self):
        if self.report is not None:
            warnings.warn("Report should be only created once")
        self.report = {key: average_helper(val, self.data_size) for key, val in self.loss_dict.items()}
        del self.loss_dict

        return self.report
