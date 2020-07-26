"""
FakeLoader and its helper functions
"""

from typing import Dict, Iterable, Iterator, List, Tuple, Union

from torch.utils.data import DataLoader, Dataset

class MyStopIteration(Exception):
    """
    A custom exception that helps dealing with complex iteration
    
    An example is shown in FakeLoader
    """
    ...

class FakeLoader:
    """
    A class that mimics a single dataloader but provides data from multiple dataloader

    I find it quite usefull in semi-supervised learning and multitask learning.
    Usually, the dataloaders in a fakeloader are expected to be based on the same dataset.
    A typical case is a fakeloader composed of a labeled dataloader and an unlabeled one.

    Dataloaders within fakeloader can be closed. The __next__ method of closed ones will
    not be called and None is returned instead.
    """
    dataloaders: Tuple[DataLoader, ...]
    tags: List[str]
    gate: Dict[DataLoader, bool]
    def __init__(self, *dataloaders, tags=None, mimic_idx=0):
        self.dataloaders = dataloaders
        self.mimic_idx = mimic_idx
        self.loader_iters = None
        self.tags = tags if tags is not None else [str(i) for i in range(len(dataloaders))]

        self._itering = False
        self.iter_set = set()

        self.init_gate()

    def init_gate(self):
        if self._itering:
            raise ValueError("You cannot init gate during iteration for now")
        self.gate = {dl: True for dl in self.dataloaders}
        self.iter_set.clear()

    def close_by_dl(self, dl):
        if self._itering:
            raise ValueError("You cannot close a gate during iteration for now")

        self.gate[dl] = False
        if not any(self.gate.values()):
            raise ValueError("At least one of the dataloader should be open!")

    def close_by_idx(self, idx):
        self.close_by_dl(self.dataloaders[idx])
    
    def close_by_tag(self, tag):
        self.close_by_idx(self.tags.index(tag))

    def named_dataloaders(self):
        return zip(self.tags, self.dataloaders)

    @property
    def dataset(self) -> Dataset:
        return self.dataloaders[self.mimic_idx].dataset

    @property
    def datasets(self) -> Iterator[Dataset]:
        return (dl.dataset for dl in self.dataloaders)

    def gen_iterator(self, dl):
        self.loader_iters[dl] = iter(dl)

    def __iter__(self):
        self._itering = True
        self.loader_iters = {dl: iter(dl) for dl in self.dataloaders}
        return self
    
    def single_next(self, dl):
        if not self.gate[dl]:
            self.iter_set.add(dl)
            return None
        try:
            dl_next = next(self.loader_iters[dl])
        except StopIteration:
            # restart or raise exception
            self.iter_set.add(dl)
            if len(self.iter_set) < len(self.dataloaders):
                self.gen_iterator(dl)
                dl_next = next(self.loader_iters[dl])
            else:
                # Here, if python builtin StopIteration is used, 
                # it will influence the generator in __next__, 
                # a custom exception is a good solution.
                raise MyStopIteration
        return dl_next

    def __next__(self):
        try:
            return tuple(self.single_next(dl) for dl in self.dataloaders)
        except MyStopIteration:
            self._itering = False
            self.iter_set.clear()
            raise StopIteration


GeneralDataLoder = Union[DataLoader, FakeLoader]

def data_train_eval_switch(dataloaders: Union[GeneralDataLoder, Iterable[GeneralDataLoder]], train: bool):
    if isinstance(dataloaders, FakeLoader):
        for ds in dataloaders.datasets:
            ds.training = train
    elif isinstance(dataloaders, DataLoader):
        ds = dataloaders.dataset
        ds.training = train
    else:
        for dl in dataloaders:
            data_train_eval_switch(dl, train)
