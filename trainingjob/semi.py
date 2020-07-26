"""
Helpers for semi-supervised laerning
"""
from .para_scheduler import FakeLoaderScheduler, SchedStepMode


class SemiJobMixin:
    """
    A mixin for semi supervised job
    """
    unlabel_tag = "UnLabeled"
    def init_loader_sched(self, para_dict):
        open_epoch = para_dict.pop("open_epoch", -1)
        train_loader = self.train_loader
        if open_epoch >= 0:
            trainloader_sched = FakeLoaderScheduler(train_loader, open_epoch) 
            train_loader.close_by_tag(self.unlabel_tag)
        else:
            trainloader_sched = None
        self._append_sched(SchedStepMode.epoch, 'unlabel_loader_scheduler', trainloader_sched)
