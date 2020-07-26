"""
Utils that help visualize in training
"""

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

matplotlib.use('Agg')

def create_heatmap(heatmap: torch.Tensor, 
                   img: torch.Tensor) -> torch.Tensor:
    """Mix heatmap and origin img for display

    Args:
        heatmap (torch.Tensor): heatmap
        img (torch.Tensor): original image

    Returns:
        torch.Tensor: mixed image
    """

    if heatmap.shape != img.shape[-2:]:
        heatmap = F.interpolate(heatmap.unsqueeze(0), img.shape[-2:], mode="bilinear")[0]

    img = img.permute(1, 2, 0).cpu().numpy()
    heatmap = 1-(heatmap+0.01)*2.0
    heatmap = F.hardtanh(heatmap, 0, 1, inplace=True)
    heatmap = heatmap.squeeze().cpu().numpy()
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    # im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax)
    fmt = '.2f' if normalize else 'd'
    sns.heatmap(cm, cmap=cmap, annot=True, ax=ax, fmt=fmt,
                xticklabels=classes, yticklabels=classes)
    # We want to show all ticks...
    ax.set(
        #    xticks=np.arange(cm.shape[1]),
        #    yticks=np.arange(cm.shape[0]),
        #    # ... and label them with the respective list entries
        #    xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label',
           xlim=(0, cm.shape[1]),
           ylim=(cm.shape[0], 0))

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     ax.text(j, i, format(cm[i, j], fmt),
    #             ha="center", va="center",
    #             color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig
    

def plot_corrcoef(features):
    sns.set()

    fig, ax = plt.subplots()
    corr = np.abs(np.corrcoef(features, rowvar=False))
    sns.heatmap(corr, annot=False, xticklabels=False, yticklabels=False, 
                ax=ax, vmin=0, vmax=1, cmap='Blues')

    ax.set(xlim=(0, corr.shape[1]),
           ylim=(corr.shape[0], 0))

    return fig
