"""
Model process aims tor alliviate the burden of heavy model definition
Model class should only define its tyical behaviour.
When a model works with different model processes, it becomes a model with focused 
functionality.

A typical usage is to define the model as a backbone which usually serves as a fully supervised classfication model.
When as semi-supervised model is needed, simply use the model with a 'semi' model process.
"""

from abc import ABCMeta, abstractmethod

class AbstractModelProcess(metaclass=ABCMeta):
    """
    An abstract class for class based model process

    This is usefull especially when model behaves differently in training and validation.
    Otherwise, define the model process as function is good enough.
    """

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def train_process(self, *args, **kwargs):
        ...

    @abstractmethod
    def valid_process(self, *args, **kwargs):
        ...


def standard_model_process(model):
    """
    Creates the most common model process i.e. ignoring label and process data
    WARNING: only the first element is considered data
    """
    def wrapped(sample_batched):
        data, *_ = sample_batched
        return model(data)
    wrapped.__name__ = "standard_model_process"
    return wrapped
