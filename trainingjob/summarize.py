import importlib
import logging
import math
import pprint
import weakref
from typing import (Any, Callable, Dict, List, Literal, Optional, Sequence,
                    Union)

import pandas as pd
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from torch.utils.tensorboard import SummaryWriter

LoggerDestination = Literal["tb", "tune", "logger"]
MetricMode = Literal["max", "min"]

class BestMetric:
    
    def __init__(self, 
                 value: Optional[float], 
                 mode: MetricMode,
                 best_step: Optional[int]):
        self.value = value
        self.mode = mode
        self.best_step = best_step

    def update_value(self, newvalue: Union[float, torch.Tensor], step: int) -> bool:
        if (self.value is None or
            self.mode == 'max' and newvalue > self.value or
            self.mode == 'min' and newvalue < self.value):
            self.value = newvalue
            self.best_step = step
            return True
        return False

    def steps_since_last_best(self, step: int) -> int:
        return step - self.best_step

class ValidationReport:
    """
    Helper class to arrange validation report 

    It is based on dataframe
    """

    def __init__(self, 
                 sched_metric_tag: str, 
                 sched_data_tag: str="Valid"):
        self.sched_metric_tag = sched_metric_tag
        self.sched_data_tag = sched_data_tag
        self.clear_cases()

    def clear_cases(self):
        self.scalars_report_cases = []
        self.figures_report_cases = []

    def clear_report(self):
        self.report = None

    def update(self, 
               case_reports: Sequence[Dict[str, Any]],
               data_tag: str):
        case_dict = {'scalars': {}, 'figures': {}}
        for case_rpt in case_reports:
            case_dict['scalars'].update(case_rpt.get('scalars', case_rpt))
            case_dict['figures'].update(case_rpt.get('figures', {}))
            
        self.scalars_report_cases.append(pd.Series(case_dict['scalars'], name=data_tag))
        self.figures_report_cases.append(pd.Series(case_dict['figures'], name=data_tag))

    def close(self):
        self.scalars_df = pd.DataFrame(self.scalars_report_cases)
        self.figures_df = pd.DataFrame(self.figures_report_cases)

        self.clear_cases()

    @property
    def sched_metric(self) -> Union[float, torch.Tensor]:
        return self.scalars_df.loc[self.sched_data_tag, self.sched_metric_tag]

    def scalars_report(self, data_tag: str=None) -> Dict:
        if data_tag is None:
            return self.scalars_df.to_dict()
        return self.scalars_df.loc[data_tag].to_dict()

    def figures_report(self) -> Dict:
        return self.figures_df.to_dict()

class Summarizer:
    """
    Generic summerizer
    provide same api for logger, tensorboard writer, tune track
    """
    best_dict: Dict[str, BestMetric]
    tracked_vars: Dict[str, Callable]
    writer: SummaryWriter
    track: Optional[Any]

    def __init__(self, 
                 writer: SummaryWriter, 
                 logger: logging.Logger,
                 best_metric: Dict[str, MetricMode],
                 tracked_vars: Dict[str, Union[torch.Tensor, Callable]],
                 with_tune: bool = False,
                 best_metric_callback: Callable[[str], None] = None):
        self.writer = writer
        self.logger = logger
        self.best_dict = {key: BestMetric(None, mode, None) for key, mode in best_metric.items()}

        self.track = None
        if with_tune:
            tune_mod = importlib.import_module(".tune", "ray")
            self.track = tune_mod.track

        self.init_tracked_vars(tracked_vars)
        self.best_metric_callback = best_metric_callback
    
    def init_tracked_vars(self, 
                          tracked_vars: Dict[str, Union[torch.Tensor, Callable]]):
        self.tracked_vars = {
            key: weakref.ref(var) if isinstance(var, torch.Tensor) else var for key, var in tracked_vars.items()
        }

    # scalers

    def _tb_add_scalars(self, 
                        scalars: dict, 
                        step: Optional[int], 
                        filter_nan: bool=True, 
                        prefix: str=None):
        for key, val in scalars.items():
            if isinstance(val, dict):
                if prefix is not None:
                    key = prefix + '/' + key
                self.writer.add_scalars(
                    key, val, step
                )
            else:
                if val is None or filter_nan and math.isnan(val):
                    continue
                if prefix is not None:
                    key = prefix + '/' + key
                self.writer.add_scalar(
                    key, val, step
                )

    def _tune_add_scalars(self, 
                          scalars: dict, 
                          step: Optional[int]):
        if self.track is None:
            return
        def tensor_to_float(x):
            if isinstance(x, torch.Tensor):
                return x.item()
            return x
        scalars = {key: tensor_to_float(val) for key, val in scalars.items()}
        self.track.log(**scalars)

    def _logger_add_scalars(self, 
                            scalars: dict, 
                            step: Optional[int]):
        self.logger.info(pprint.pformat(scalars))

    def add_scalars(self, 
                    scalars: dict, 
                    step: int=None, 
                    *, 
                    destination: LoggerDestination="tb", 
                    update_best: bool=False, 
                    append_tracked: bool=False, 
                    filter_nan: bool=True,
                    prefix: str=None):

        if append_tracked:
            scalars.update(
                self.get_tracked_state()
            )
        if update_best:
            best_rpt = self.update_best(scalars, step)
            scalars.update(best_rpt)
        if destination == "tb":
            self._tb_add_scalars(scalars, step, filter_nan, prefix)
        elif destination == "tune":
            self._tune_add_scalars(scalars, step)
        elif destination == "logger":
            self._logger_add_scalars(scalars, step)
        
    def add_figures(self, 
                    figs: Dict[str, Union[Figure, List[Figure]]],
                    step: int):
        for tag, fig in figs.items():
            if isinstance(fig, dict):
                for sub_tag, sub_fig in fig.items():
                    self.writer.add_figure(tag+'/'+sub_tag, sub_fig, step)
            else:
                self.writer.add_figure(tag, fig, step)
    
    def update_best(self, 
                    metrics:Dict[str, float], 
                    step: int):
        for name, newvalue in metrics.items():
            if name in self.best_dict:
                metric = self.best_dict[name]
                if metric.update_value(newvalue, step) and self.best_metric_callback is not None:
                    self.best_metric_callback(name)

        return {"best_" + key: val.value for key, val in self.best_dict.items()}

    def get_tracked_state(self):
        tracked_state = {key: val() for key, val in self.tracked_vars.items() }
        return {key: val for key, val in tracked_state.items() if val is not None}

    def update_tracked(self, 
                       step: int):
        tracked_state = self.get_tracked_state()
        self.add_scalars(tracked_state, step)

    # text
    def _logger_add_text(self, 
                         tag: str, 
                         text_string: str, 
                         step: int=None):
        self.logger.info(tag+': '+text_string)

    def _tb_add_text(self, 
                     tag: str, 
                     text_string: str, 
                     step: int=None):
        self.writer.add_text(tag, text_string, step)

    def add_text(self, 
                 tag: str, 
                 text_string: str, 
                 step: int=None, 
                 destination: LoggerDestination="tb"):
        if destination == "tb":
            self._tb_add_text(tag, text_string, step)
        elif destination == "logger":
            self._logger_add_text(tag, text_string, step)
            
    # others
    def add_weight_histogram(self, 
                             model: nn.Module, 
                             step: int):
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                tag = name.replace('.', '/')
                try:
                    self.writer.add_histogram(tag, module.weight.view((-1,)), step)
                except ValueError as e:
                    self.logger.error(f"{e} when histgram {tag}")

    def add_valid_report(self, 
                         report: ValidationReport, 
                         step: int, 
                         *, 
                         scalar_prefix: str=None):
        scalars_report = report.scalars_report()
        self.add_scalars(scalars_report, step, prefix=scalar_prefix)
        figures_report = report.figures_report()
        self.add_figures(figures_report, step)
