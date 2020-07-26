import functools
from abc import ABC
from copy import copy
from typing import Callable, Iterator, Sequence, Tuple, Union

import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from .ema import EMA, ema_autoswap
from .functional.data_hook import data_hook_split
from .functional.generic import *
from .loss import ModernLoss
from .metrics import ClassificationMetric
from .model_process import standard_model_process
from .para_scheduler import AbstractScheduler
from .summarize import Summarizer, ValidationReport
from .training_job import TrainingJob, get_checkpoint_stop_hook

__all__ = ["CommonClsJob"]


class ClsJobMixin(ABC):

    def data_hook(self, sample_batched):
        data, label = sample_batched
        if len(data.size()) == 3:
            data.unsqueeze_(1)
        return data.to(self.device), label.to(self.device)

    def accuracy_hook(self, output, label):
        pred = output.max(1)[1]
        num_correct = pred.eq(label).sum().item()
        cm = confusion_matrix(label.to('cpu').numpy(), pred.to('cpu').numpy(),
                               labels=list(range(self.num_classes)))

        return num_correct, cm

    
    def grad_cam(self, mode='bilinear', with_relu=False, positive_only=True):
        """
        Get gradcam 
        for visualization only
        """

        self.model.eval()

        for sample in self.test_loader:
            data, label = self.data_hook(sample)
            heatmap = self.model.batch_grad_cam(data, label, mode=mode, with_relu=with_relu, positive_only=False)
                
            if hasattr(self.test_loader.dataset, "denormalize"):
                data = self.test_loader.dataset.denormalize(data)

            yield data.squeeze().numpy(), heatmap.squeeze().numpy()

class CommonClsJob(ClsJobMixin, TrainingJob):
    CONFIG_SOURCE = 'Classifier'

    def init_defaults(self):
        self.reccls_defaults = {
            'epochs': self.epochs,
            'train_loader': self.train_loader,
            'test_loader': self.test_loader,
            'summarizer': self.summarizer,
            'report_helper': self.report_helper,
            'batch_counter': self.batch_counter,
            'epoch_counter': self.epoch_counter,
            'model': self.model,
            'data_hook': self.data_hook,
            'metrics': self.metrics,
            'opt': self.opt,
            'epoch_schedulers': self.epoch_schedulers.values(),
            'iteration_schedulers': self.iteration_schedulers.values(),
            'checkpoint_stop_hook': get_checkpoint_stop_hook(self),
            'loss': self.loss
        }

    def best_metric(self):
        return {"ClsLoss": "min", "Accuracy": "max"}

    def init_loss(self, para_dict):
        self.loss = ModernLoss(nn.CrossEntropyLoss(), "ClsLoss")

    def init_metrics(self):
        self.metrics = ClassificationMetric(model_hook=lambda x: x, 
                                            accurcy_hook=self.accuracy_hook, 
                                            classes=self.test_loader.dataset.classes)
        
    def run(self):
        para_dict = copy(self.reccls_defaults)

        return self._run(para_dict)

    def _run(self, para_dict):
        extra_hook = functools.partial(
            mri_img_hook, 
            writer= self.summarizer.writer,
            add_image_hook=self.test_loader.dataset.writer_add_image
        )

        return self.train_reccls(**para_dict, model_process=standard_model_process(self.model), extra_hook=extra_hook)

    @staticmethod
    def train_reccls(epochs: int, 
                     train_loader: DataLoader,
                     test_loader: DataLoader,
                     model: nn.Module, 
                     opt: optim.Optimizer,
                     summarizer: Summarizer,
                     batch_counter: Iterator[int],
                     *,
                     report_helper,
                     epoch_schedulers: Sequence[AbstractScheduler],
                     iteration_schedulers: Sequence[AbstractScheduler],
                     data_hook: Union[Callable, Tuple[Callable, Callable]] = lambda x: x,
                     metrics = None,
                     loss,
                     epoch_counter: Iterator[int],
                     model_process,
                     extra_hook=None,
                     checkpoint_stop_hook):
        """
        TODO Split this into several parts
        """
        train_data_hook, test_data_hook = data_hook_split(data_hook)

        epoch_train = get_epoch_train(
            data_loader = train_loader,
            model=model,
            model_process = model_process,
            opt = opt,
            loss = loss.from_model,
            schedulers = iteration_schedulers,
            summarizer = summarizer,
            batch_counter = batch_counter,
            data_hook = train_data_hook
        )

        epoch_valid = get_valid(
            data_loaders={
                "Train": train_loader,
                "Valid": test_loader,
            },
            model = model,
            model_process = model_process,
            loss = loss.from_model,
            data_hook=test_data_hook,
            summarizer=summarizer,
            loss_tags=loss.tags,
            metrics=metrics,
            extra_hook=extra_hook,
            report_helper=report_helper
        )

        if isinstance(opt, EMA):
            epoch_valid = ema_autoswap(opt)(epoch_valid)

        return generic_train(epochs, 
                             model=model,
                             summarizer=summarizer,
                             epoch_counter=epoch_counter,
                             schedulers=epoch_schedulers,
                             epoch_valid=epoch_valid,
                             epoch_train=epoch_train,
                             checkpoint_stop_hook = checkpoint_stop_hook,
                             report_helper=report_helper
                             )

class TrainOnlyClsJob(CommonClsJob):
    def init_metrics(self):
        self.metrics = ClassificationMetric(model_hook=lambda x: x, accurcy_hook=self.accuracy_hook, classes=self.train_loader.dataset.classes)

    def init_report_helper(self):
        self.report_helper = ValidationReport("Accuracy", "Train")

    def _run(self, para_dict):

        return self.train_reccls(**para_dict, model_process=standard_model_process(self.model))

    @staticmethod
    def train_reccls(epochs: int, 
                     train_loader: DataLoader,
                     test_loader: DataLoader,
                     model: nn.Module, 
                     opt: optim.Optimizer,
                     summarizer: Summarizer,
                     batch_counter: Iterator[int],
                     *,
                     report_helper,
                     epoch_schedulers: Sequence[AbstractScheduler],
                     iteration_schedulers: Sequence[AbstractScheduler],
                     data_hook: Union[Callable, Tuple[Callable, Callable]] = lambda x: x,
                     metrics = None,
                     loss,
                     epoch_counter: Iterator[int],
                     model_process,
                     extra_hook=None,
                     checkpoint_stop_hook):
        train_data_hook, test_data_hook = data_hook_split(data_hook)

        epoch_train = get_epoch_train(
            data_loader = train_loader,
            model=model,
            model_process = model_process,
            opt = opt,
            loss = loss.from_model,
            schedulers = iteration_schedulers,
            summarizer = summarizer,
            batch_counter = batch_counter,
            data_hook = train_data_hook
        )

        epoch_valid = get_valid(
            data_loaders={
                "Train": train_loader,
            },
            model = model,
            model_process = model_process,
            loss = loss.from_model,
            data_hook=test_data_hook,
            summarizer=summarizer,
            loss_tags=loss.tags,
            metrics=metrics,
            extra_hook=extra_hook,
            report_helper=report_helper,
            hp_source_tag="Train"
        )

        if isinstance(opt, EMA):
            epoch_valid = ema_autoswap(opt)(epoch_valid)

        return generic_train(epochs, 
                             model=model,
                             summarizer=summarizer,
                             epoch_counter=epoch_counter,
                             schedulers=epoch_schedulers,
                             epoch_valid=epoch_valid,
                             epoch_train=epoch_train,
                             checkpoint_stop_hook = checkpoint_stop_hook,
                             report_helper=report_helper
        )
