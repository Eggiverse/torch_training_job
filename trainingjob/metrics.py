import warnings
from abc import ABCMeta, abstractmethod
from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_curve

from .functional.visualization import plot_confusion_matrix


def average_helper(val, size):
    """
    Ensure program runs even if size is zero
    """
    try:
        return val / size
    except ZeroDivisionError:
        warnings.warn("Datasize seems to be 0!")
        return -0.1

class AbstractMetric(metaclass=ABCMeta):
    @abstractmethod
    def from_model(self, model_output, batch_sample):
        ...
    @abstractmethod
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        ...

class ClassificationMetric(AbstractMetric):
    """
    Helper class for accuracy, auc, confusion matrics and so on
    """

    def __init__(self, accurcy_hook=None, model_hook=lambda x: x, classes=None):
        self.accuracy_hook = accurcy_hook
        self.model_hook = model_hook
        self.classes = classes
        self.init_list()
    
    def init_list(self):
        self.label_list = []
        self.pred_list = []
        
    def from_model(self, model_output, batch_sample):
        pred = self.model_hook(model_output)
        _, label = batch_sample
        self.append(pred, label)

    def append(self, pred, label):
        self.label_list.append(label)
        self.pred_list.append(pred)

    def get_metrics(self):
        label_list = torch.cat(self.label_list)
        pred_list = torch.cat(self.pred_list)
        self.init_list()
        num_correct, cm = self.accuracy_hook(pred_list, label_list)
        accuracy = average_helper(num_correct, len(label_list))
        cm = plot_confusion_matrix(cm, self.classes)
        return {
            'scalars': {
                "Accuracy": accuracy,
            },
            'figures': {
                "CM": cm
            }
        }


def get_roc(labels, predictions):
    labels = np.squeeze(labels)
    predictions = np.squeeze(predictions)
    labels = labels.astype(int)
    return roc_curve(labels, predictions)

def get_acc_cm(labels, predictions, fpr, tpr, threshold):
    optimal_index = np.argmax(tpr-fpr)
    optimal_threshold = threshold[optimal_index]
    predicted_labels = (predictions > optimal_threshold).astype(int)

    correct_num = (predicted_labels == labels).sum()
    acc = correct_num / labels.size

    cm = confusion_matrix(labels, predicted_labels, labels=[0, 1])

    return acc, cm

