"""
Most generic training fuctions
"""

import itertools
from typing import Any, Callable, Dict, Iterable, Iterator, Sequence, Tuple

import more_itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..data.fakeloader import GeneralDataLoder, data_train_eval_switch
from ..loss import MultiLoss
from ..metrics import AbstractMetric
from ..para_scheduler import AbstractScheduler, scheduler_step
from ..summarize import Summarizer, ValidationReport

__all__ = [
    'generic_train',
    'get_epoch_train',
    'get_valid'
]

def enumslice(iterable: Iterable, stop: int) -> Iterable[Tuple[int, Any]]:
    return enumerate(itertools.islice(iterable, stop))

def generic_train(
    epochs: int,
    *,
    model: nn.Module,
    summarizer: Summarizer,
    epoch_counter: Iterator[int],
    schedulers: Iterable[AbstractScheduler],
    epoch_valid: Callable,
    epoch_train: Callable,
    checkpoint_stop_hook: Callable[..., None] = None,
    report_helper: ValidationReport
):
    if epochs == 0:
        return

    current_epoch_list, epoch_counter = more_itertools.spy(epoch_counter)
    current_epoch = current_epoch_list[0]

    epoch_valid(current_epoch-1, False)

    for phase_epoch, epoch in enumslice(epoch_counter, epochs):
    
        epoch_train(epoch)

        epoch_valid(epoch, True)

        for sched in schedulers:
            scheduler_step(sched, report_helper.sched_metric)

        if checkpoint_stop_hook is not None:
            checkpoint_stop_hook(phase_epoch, epoch, report_helper)

        report_helper.clear_report()


def get_epoch_train(
    *,
    data_loader: GeneralDataLoder,
    model: nn.Module,
    model_process: Callable = None,
    opt: optim.Optimizer,
    loss: Callable[..., MultiLoss],
    schedulers : Iterable[AbstractScheduler],
    summarizer: Summarizer,
    batch_counter: Iterator[int],
    data_hook: Callable = lambda x: x
) -> Callable[[int], Any]:
    return lambda epoch: generic_epoch_train(
        epoch,
        data_loader = data_loader,
        model = model,
        model_process = model_process,
        opt = opt,
        loss = loss,
        schedulers = schedulers,
        summarizer = summarizer,
        batch_counter = batch_counter,
        data_hook = data_hook
    )


def generic_epoch_train(
    epoch: int,
    *,
    data_loader: GeneralDataLoder,
    model: nn.Module,
    model_process: Callable = None,
    opt: optim.Optimizer,
    loss: Callable[..., MultiLoss],
    schedulers,
    summarizer: Summarizer,
    batch_counter: Iterator[int],
    data_hook: Callable = lambda x: x
):
    """
    Define the process of training an epoch
    """
    model.train()
    data_train_eval_switch(data_loader, True)
    if model_process is None:
        model_process = model

    for batch_idx, sample_batched in enumerate(data_loader):
        writer_step = next(batch_counter)

        sample_batched = data_hook(sample_batched)

        output = model_process(sample_batched)

        batch_loss = loss(output, sample_batched)

        opt.zero_grad()
        batch_loss.backward()
        opt.step()

        batch_rpt = {f"Batch/{key}": val for key, val in batch_loss.loss_dict.items()}
        summarizer.add_scalars(batch_rpt, writer_step)

        for sched in schedulers:
            scheduler_step(sched, batch_loss.sched_metric)

def get_batchsize(sample_batched):
    # TODO support fake batch
    data, *_ = sample_batched
    if isinstance(data, list):
        data = data[0]
    return data.size(0)

def get_valid(
    *,
    data_loaders: Dict[str, DataLoader],
    model: nn.Module,
    model_process: Callable = None,
    loss: Callable,
    data_hook: Callable = lambda x: x,
    summarizer: Summarizer,
    loss_tags: Sequence[str],
    metrics: AbstractMetric,
    extra_hook: Callable = None,
    hp_source_tag: str = "Valid",
    report_helper: ValidationReport
) -> Callable[[int, bool], Any]:
    return lambda epoch, with_tune: generic_valid(
        epoch, with_tune,
        data_loaders = data_loaders,
        model = model,
        model_process = model_process,
        loss = loss,
        data_hook = data_hook,
        summarizer = summarizer,
        loss_tags = loss_tags,
        metrics = metrics,
        extra_hook = extra_hook,
        hp_source_tag = hp_source_tag,
        report_helper = report_helper
    )

def generic_valid(
    epoch: int,
    with_tune: bool = True,
    *,
    data_loaders: Dict[str, DataLoader],
    model: nn.Module,
    model_process: Callable = None,
    loss: Callable,
    data_hook: Callable = lambda x: x,
    summarizer: Summarizer,
    loss_tags: Sequence[str],
    metrics: AbstractMetric,
    extra_hook: Callable = None,
    hp_source_tag: str = "Valid",
    report_helper: ValidationReport
):
    if model_process is None:
        model_process = model

    data_train_eval_switch(data_loaders.values(), False)

    def get_metric(data_loader: GeneralDataLoder, extra_tag: str=""):
        loss_accumulater = MultiLoss.getAccumulater(loss_tags)

        for batch_idx, sample_batched in enumerate(data_loader):

            sample_batched = data_hook(sample_batched)

            output = model_process(sample_batched)

            batchsize = get_batchsize(sample_batched)

            # get loss
            batch_loss = loss(output, sample_batched)
            # get other metrics
            metrics.from_model(output, sample_batched)
            # something like making plots or show images
            if extra_hook is not None:
                extra_hook(output, sample_batched, batch_idx, extra_tag, epoch)


            loss_accumulater.accum(batch_loss, batchsize=batchsize)

        loss_rpt = loss_accumulater.get_average()
        metric_rpt = metrics.get_metrics()

        return loss_rpt, metric_rpt

    model.eval()

    with torch.no_grad():

        summarizer.add_weight_histogram(model, epoch)

        for tag, data_loader in data_loaders.items():
            report_helper.update(get_metric(data_loader, tag), tag)
        report_helper.close()

        summarizer.add_valid_report(report_helper, epoch, scalar_prefix="Epoch")

        if with_tune:
            summarizer.add_scalars(report_helper.scalars_report(hp_source_tag),
                                   destination="tune", 
                                   append_tracked=True, 
                                   update_best=True)
