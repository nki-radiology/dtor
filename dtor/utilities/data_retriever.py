# -*- coding: utf-8 -*-
"""Process our data choice"""

__author__ = "Sean Benson"
__copyright__ = "MIT"

from dtor.datasets.dataset_nominal import CTImageDataset
from dtor.datasets.dataset_mnist import MNIST3DDataset
from torchvision import transforms
from torchvision.transforms import transforms
import torch
import torch.nn as nn


class ConvertCDHWtoDCHW(nn.Module):
    """Convert tensor from (C, D, H, W) to (D, C, H, W)
    """

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(1, 0, 2, 3)


class ConvertDCHWtoCDHW(nn.Module):
    """Convert tensor from (D, C, H, W) to (C, D, H, W)
    """

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(1, 0, 2, 3)


def get_data(name, csv_loc=None, fold=None, aug=False,
             mean=None,
             std=None,
             dim=3):
    """
    Helper function for datasets
    Args:
        name: dataset name
        csv_loc: location of file locs
        fold: which fold
        aug: do you want to include augmentations
        mean: mean of the channels for transform
        std: std deviation of the channels for the transform

    Returns: torch dataset

    """

    # Do not throw randomised transforms in for the case of evaluation
    tr_eval = [
        ConvertCDHWtoDCHW(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=mean, std=std),
        ConvertDCHWtoCDHW()
    ]
    tr_aug = [
        ConvertCDHWtoDCHW(),
        transforms.RandomAffine(0),
        transforms.GaussianBlur(3),
        transforms.RandomRotation(degrees=5),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=mean, std=std),
        ConvertDCHWtoCDHW()
    ]

    if dim == 2:
        tr_eval = tr_eval[1:-1]
        tr_aug = tr_aug[1:-1]

    tr_eval = transforms.Compose(tr_eval)
    tr_aug = transforms.Compose(tr_aug)

    assert name.lower() in ["mnist3d", "ltp"]
    if name.lower() == "mnist3d":
        train_ds = MNIST3DDataset(h5_file="data/external/mnist/full_dataset_vectors.h5",
                                  tr_test="train",
                                  transform=tr_aug if aug else tr_eval)
        val_ds = MNIST3DDataset(h5_file="data/external/mnist/full_dataset_vectors.h5",
                                tr_test="test", transform=tr_eval)
    else:

        train_ds = CTImageDataset(chunked_csv=csv_loc,
                                  fold=fold,
                                  tr_test="train",
                                  transform=tr_aug if aug else tr_eval,
                                  dim=dim
                                  )
        val_ds = CTImageDataset(chunked_csv=csv_loc,
                                fold=fold,
                                tr_test="test",
                                transform=tr_eval,
                                dim=dim
                                )

    return train_ds, val_ds
