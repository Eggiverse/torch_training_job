"""
Datahook aims to serve between dataloader and model input.

Typically, it moves data to gpu.
"""

import torch
from typing import Callable, Sequence, Tuple, Union


def common_data_hook(device):
    def data_hook(sample_batched):
        data, label = sample_batched
        if data.ndim == 3:
            data.unsqueeze_(1)
        return data.to(device, non_blocking=True), label.to(device, non_blocking=True)
    return data_hook


def mixed_data_hook(data_hook):
    def wrapped_hook(sampled):
        """
        A datahook that deals with semi-labeled sample
        """
        labeled_sample, unlabeled_sample = sampled
        if unlabeled_sample is not None:
            (label_data, label_target) = labeled_sample
            (unlabel_data, unlabel_target) = unlabeled_sample

            data = torch.cat((label_data, unlabel_data), dim=0)
            fake_unlabel_target = torch.full_like(unlabel_target, -1)
            target = torch.cat((label_target, fake_unlabel_target))
            mixed_sample = (data, target)
        else:
            mixed_sample = labeled_sample

        mixed_sample = data_hook(mixed_sample)

        return mixed_sample
    return wrapped_hook

def multi_data_hook(device):
    def wrapped_hook(sample_batched):
        data, label = sample_batched
        def process_onecase(d):
            if d.ndim == 3:
                d.unsqueeze_(1)
            return d.to(device, non_blocking=True)
        data = [process_onecase(d) for d in data]
        return data, label.to(device, non_blocking=True)
    return wrapped_hook

def mri_data_hook(data_hook: Callable, batch_transform: Callable, extra_process: Callable = None):
    """A decorator that makes a typical data hook serve mri style data

    It aims to apply batch_transform and extra_process (usually normalization)

    Args:
        data_hook (Callable): The decorated data hook
    """

    def wrapped_hook(sampled):
        data, label = data_hook(sampled)
        if isinstance(data, (list, tuple)):
            data = [batch_transform(d) for d in data]
        else:
            data = batch_transform(data)
        if extra_process is not None:
            data = extra_process(data)
        return data, label
    return wrapped_hook

def _transform_applier(transform: Callable[[torch.Tensor], torch.Tensor],
                       extra_process: Callable[[torch.Tensor], torch.Tensor],
                       data: Union[Sequence[torch.Tensor], torch.Tensor]
                       ) -> Union[Sequence[torch.Tensor], torch.Tensor]:
    """Help apply transform on either one or a group of images

    Args:
        transform (Callable[[torch.Tensor], torch.Tensor]): [description]
        data (Union[Sequence[torch.Tensor], torch.Tensor]): [description]

    Returns:
        Union[Sequence[torch.Tensor], torch.Tensor]: [description]
    """

    if not isinstance(data, (tuple, list)):
        data = transform(data)
    else:
        data = [transform(d) for d in data]
    if extra_process is not None:
        data = extra_process(data)
    return data

def _cat_helper(data):
    if isinstance(data[0], (tuple, list)):
        return list(map(lambda x: torch.cat(tuple(x)), zip(*data)))
    
    return torch.cat(data)

def mri_mixed_data_hook(data_hook: Callable,
                        transforms: Sequence[Callable],
                        extra_processes: Sequence[Callable]) -> Callable:
    labeled_transform, weak_transform, strong_transform = transforms
    labeled_extra, weak_extra, strong_extra = extra_processes
    def wrapped_hook(sampled):
        labeled_sample, unlabeled_sample = sampled
        labeled_sample = data_hook(labeled_sample)

        labeled_data, labeled_target = labeled_sample

        labeled_data = _transform_applier(labeled_transform, labeled_extra, labeled_data)

        if unlabeled_sample is None:
            return labeled_data, labeled_target
        
        # Unlabeled sample may contain or not contain label
        # Ignore label if exists
        # WARNING: it won't work if two views are provided
        # TODO fix two view problem
        if isinstance(unlabeled_sample, (tuple, list)) and len(unlabeled_sample) == 2:
            unlabeled_data, _ = unlabeled_sample
        else:
            unlabeled_data = unlabeled_sample

        try:
            unlabeled_length = unlabeled_data.size(0)
        except AttributeError:
            unlabeled_length = unlabeled_data[0].size(0)


        fake_unlabel_target_weak = torch.full([unlabeled_length], -1, dtype=labeled_target.dtype)
        fake_unlabel_target_strong = torch.full([unlabeled_length], -2, dtype=labeled_target.dtype)

        unlabeled_data_weak, fake_unlabel_target_weak = data_hook((unlabeled_data, fake_unlabel_target_weak))
        unlabeled_data_strong, fake_unlabel_target_strong = data_hook((unlabeled_data, fake_unlabel_target_strong))

        unlabeled_data_weak = _transform_applier(weak_transform, weak_extra, unlabeled_data_weak)
        unlabeled_data_strong = _transform_applier(strong_transform, strong_extra, unlabeled_data_strong)


        outdata = _cat_helper((labeled_data, unlabeled_data_strong, unlabeled_data_weak))
        outlabel = _cat_helper((labeled_target, fake_unlabel_target_strong, fake_unlabel_target_weak))
        return outdata, outlabel
    return wrapped_hook

def concat_data_hook(data_hooks):
    def wrapped_hook(sampled):
        sampled = [dh(s) for s, dh in zip(sampled, data_hooks)]
        datas, labels = map(tuple, zip(*sampled))
        return torch.cat(datas), torch.cat(labels)
    return wrapped_hook

def data_hook_split(data_hook: Union[Callable, Tuple[Callable, Callable]]) -> Tuple[Callable, Callable]:
    """Split data_hook into train and test/valid data_hook

    Not recommended to use now
    """

    if callable(data_hook):
        return data_hook, data_hook
    elif isinstance(data_hook, tuple) and len(data_hook) == 2:
        return data_hook
    raise TypeError("Data hook should be callable")
