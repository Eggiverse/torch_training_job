import itertools
import logging
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from dict_recursive_update import recursive_update
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .data.fakeloader import GeneralDataLoder
from .ema import EMA
from .functional import *
from .functional.optimizer import get_opt
from .para_scheduler import (LR_SCHED_DICT, AbstractScheduler, SchedStepMode,
                             get_ts_scheduler)
from .step_counter import StepCounter
from .summarize import Summarizer, ValidationReport
from .utils import get_device

__all__ = [
    'TrainingJob'
]


class TrainingJob(metaclass=ABCMeta):
    """A training job

    The main logic is defined in method :py:meth:`run`

    >>> myjob = XXXTrainingJob(config)
    >>> myjob.run()
    """
    model: nn.Module
    opt: optim.Optimizer
    epoch_schedulers: Dict[str, AbstractScheduler]
    iteration_schedulers: Dict[str, AbstractScheduler]
    job_path: Path
    device: torch.device
    config: Dict[str, Any]
    # writer: SummaryWriter
    summarizer: Summarizer
    train_loader: DataLoader
    test_loader: DataLoader
    model_path: Path
    data_lock: bool = False
    CONFIG_SOURCE: str

    def __init__(self, config:Dict[str, Any], model=None, job_path:Union[str, Path]="", **kwargs):
        config.update(kwargs)

        if job_path == "":
            self.job_path = Path("runs") / config['JobName'] / str(config['JobID'])
        else:
            self.job_path = Path(job_path)

        original_config = deepcopy(config)

        # set the device for the job
        self.device = get_device(config['Cuda'])

        self.config = config

        self.epochs = config[self.CONFIG_SOURCE]["epochs"]

        # load data 
        self.init_data(config["Data"])

        self.init_model(config["Model"])

        self.init_counters()

        # For now, these use the same config source
        # TODO Improve config design

        self.hp_dict = edict()

        self.init_hyperpara(config[self.CONFIG_SOURCE])

        self.init_opt(config[self.CONFIG_SOURCE])

        self.init_loss(config[self.CONFIG_SOURCE])

        self.init_loader_sched(config[self.CONFIG_SOURCE])

        self.init_metrics()

        self.extra_inits()

        # init tensorboard writer
        self.init_summarizer(original_config)

        self.init_defaults()

        if (resume_config:=config.get("Resume", None)) is not None:
            self.resume_job(resume_config)

    def extra_inits(self):
        """extra inits for sub classes"""
        ...

    def resume_job(self, cont_config):
        strict_load = cont_config.get('strict_load', True)
        self.load_state_dict(
            torch.load(cont_config['checkpoint_path'], 
                       map_location=self.device),
                       strict=strict_load)

    def named_schedulers(self):
        return filter(lambda x: x[1] is not None, itertools.chain(self.epoch_schedulers.items(), self.iteration_schedulers.items()))

    def state_dict(self):
        return {
            'epoch': self.epoch_counter.last_step,
            'iteration': self.batch_counter.last_step,
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'scheduler': {name: sched.state_dict() for name, sched in self.named_schedulers()}
        }

    def _load_counter_state(self, state_dict):
        epoch = state_dict.get('epoch', 1)
        iteration = state_dict.get('iteration', 1)
        self.batch_counter.last_step = iteration
        self.epoch_counter.last_step = epoch

    def _load_scheduler_state(self, state_dict, strict=True):
        for name, sched in self.named_schedulers():
            try:
                sched.load_state_dict(state_dict[name])
            except KeyError:
                if strict:
                    raise KeyError(f"{name} is not included in the loaded state")
                

    def load_state_dict(self, state_dict: Dict, strict=True):
        state_dict = state_dict.copy()

        self._load_counter_state(state_dict)

        model_state = state_dict.get('model', None)
        if model_state is not None:
            self.model.load_state_dict(model_state, strict)
        
        opt_state = state_dict.get('opt', None)
        if opt_state is not None:
            self.opt.load_state_dict(opt_state)
        
        sched_state = state_dict.get('scheduler', None)
        if sched_state is not None:
            self._load_scheduler_state(sched_state, strict)

    def model_add_load_hook(self, model):
        pass

    @abstractmethod
    def get_model(self, model_config, **kwargs):
        ...

    def init_model(self, model_config):
        self.model_path = self.job_path / "Models"
        self.model_path.mkdir(parents=True, exist_ok=True)

        model = self.get_model(model_config, named_paras={'num_classes': self.num_classes})

        self.model_add_load_hook(model)

        if model is not None:
            self.model = model.to(self.device)
            if model_config['load_model']:
                strict_load = model_config.get('strict_load', True)
                self.model.load_state_dict(
                    torch.load(model_config['pre_model_path'], 
                               map_location=self.device), 
                    strict=strict_load)

    @abstractmethod
    def init_defaults(self):
        pass

    def init_counters(self):
        self.batch_counter = StepCounter(1)
        self.epoch_counter = StepCounter(1)

    @abstractmethod
    def load_data(self, data_config) -> Tuple[GeneralDataLoder, GeneralDataLoder]:
        ...

    def init_data(self, data_config):
        """
        init dataloaders
        use data lock to ensure it runs one time only
        """
        if self.data_lock:
            return
        self.data_lock = True
        self.train_loader, self.test_loader = self.load_data(data_config)
        self.num_classes = self.train_loader.dataset.num_classes

    @abstractmethod
    def best_metric(self):
        ...

    def tracked_vars(self):
        return {"lr": lambda : self.opt.param_groups[0]['lr']}

    def init_summarizer(self, original_config):
        with_tune = self.config.get('with_tune', False)

        summary_path = self.job_path / "Summaries" / str(datetime.now().strftime('%b%d_%H-%M-%S'))

        writer = SummaryWriter(log_dir=summary_path)

        logging.basicConfig()
        logger = logging.getLogger("TestLogger")
        logger.setLevel(logging.INFO)

        for key, val in original_config.items():
            writer.add_text(f"config/{key}", str(val))
        
        summarizer = Summarizer(writer, logger, 
                                best_metric=self.best_metric(),
                                tracked_vars=self.tracked_vars(),
                                with_tune=with_tune,
                                best_metric_callback=get_best_metric_callback(self))

        self.summarizer = summarizer
        self.init_report_helper()

    @abstractmethod
    def init_metrics(self):
        ...

    def _paramgroups(self, para_dict):
        """
        Return param groups for optimizer initialization
        """
        return self.model.parameters()

    def init_opt(self, para_dict):
        opt = get_opt(para_dict, self._paramgroups(para_dict))
        
        ema_decay = para_dict.get("ema_decay", None)
        if ema_decay is not None:
            self.init_para_sched(para_dict, 'ema_decay')

            opt = EMA(opt, decay=self.hp_dict.ema_decay, zero_debias=True)
        
        self.opt = opt
        
        self.init_lr_sched(opt, para_dict['lr_scheduler'])

    def _init_scheds(self):
        if not hasattr(self, "epoch_schedulers"):
            self.epoch_schedulers = {}
        if not hasattr(self, "iteration_schedulers"):
            self.iteration_schedulers = {}

    def _append_sched(self, step_mode, name, sched):
        if step_mode is SchedStepMode.epoch:
            self.epoch_schedulers[name]=sched
        elif step_mode is SchedStepMode.iteration:
            self.iteration_schedulers[name]=sched

    def init_lr_sched(self, opt: optim.Optimizer, lr_config):
        self._init_scheds()

        def get_one_sched(sched_config):

            sched_step_mode = SchedStepMode.get_mode(sched_config.pop('step_mode', 'epoch'))

            sched_type_name = sched_config.get("type", "Plateau")
            sched_name = sched_config.get("name", sched_type_name)
            sched_type = LR_SCHED_DICT[sched_type_name]
            
            sched_paras = sched_config.get("config", sched_config)
            sched = sched_type(opt, **sched_paras)

            self._append_sched(sched_step_mode, sched_name, sched)

        if isinstance(lr_config, dict):
            get_one_sched(lr_config)
        elif isinstance(lr_config, list):
            for _opt_conf in lr_config:
                get_one_sched(_opt_conf)

    def init_para_sched(self, config, para_name, para_value=None):
        if isinstance(para_name, (tuple, list)):
            if para_value is None:
                para_value = [None] * len(para_name)
            for pn, pv in zip(para_name, para_value):
                self.init_para_sched(config, pn, pv)
            return
        self._init_scheds()

        para_value = config[para_name] if para_value is None else para_value
        # print("Para value is ", para_value)
        para = torch.tensor(para_value, device=self.device)
        # config[para_name] = para
        self.hp_dict[para_name] = para

        para_sched = config.pop(f'{para_name}_scheduler', None)
        if para_sched is not None:

            sched_step_mode = SchedStepMode.get_mode(para_sched.pop('step_mode', 'epoch'))

            para_sched = get_ts_scheduler(para, para_sched)

            self._append_sched(sched_step_mode, f'{para_name}_scheduler', para_sched)

    def init_hyperpara(self, para_dict):
        pass        

    @abstractmethod
    def init_loss(self, para_dict):
        ...

    @abstractmethod
    def run(self):
        """Main logic of the job"""
        ...

    def init_loader_sched(self, para_dict):
        pass

    def init_report_helper(self):
        self.report_helper = ValidationReport("Accuracy")

    @classmethod
    def run_hpara(cls, config:Dict, hpara_config:Dict):
        recursive_update(config, hpara_config)
        job = cls(config, job_path="runs")
        job.run()

    def extra_repr(self) -> str:
        return ""

    def __repr__(self):
        return f"Job on model {self.model}\n{self.extra_repr()}"

def get_best_metric_callback(job: TrainingJob) -> Callable[[str], None]:
    def best_metric_callback(metric_name: str):
        torch.save(job.model.state_dict(), job.model_path / f"best_{metric_name}.pt")
    return best_metric_callback

def get_checkpoint_stop_hook(job: TrainingJob, 
                             early_stop: Callable[[int, int, ValidationReport], None]=None
                             ) -> Callable[[int, int, ValidationReport], None]:
    def checkpoint_stop_hook(phase_epoch: int, epoch: int, valid_rst: ValidationReport):
        torch.save(job.state_dict(), job.model_path / "job_checkpoint.pt")
        if early_stop is not None:
            early_stop(phase_epoch, epoch, valid_rst)
    return checkpoint_stop_hook
