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


def get_data(name, csv_loc=None, fold=None, aug=False):
    assert name.lower() in ["mnist3d", "ltp"]
    if name.lower() == "mnist3d":
        train_ds = MNIST3DDataset(h5_file="data/external/mnist/full_dataset_vectors.h5", tr_test="train")
        val_ds = MNIST3DDataset(h5_file="data/external/mnist/full_dataset_vectors.h5", tr_test="test")
    else:
        mean = (0.43216, 0.394666, 0.37645)
        std = (0.22803, 0.22145, 0.216989)

        # Do not throw randomised transforms in for the case of evaluation
        tr_eval = transforms.Compose([
            ConvertCDHWtoDCHW(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=mean, std=std),
            ConvertDCHWtoCDHW()
            ])
        tr_aug = transforms.Compose([
            ConvertCDHWtoDCHW(),
            transforms.RandomAffine(0),
            transforms.GaussianBlur(3),
            transforms.RandomRotation(degrees=5),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=mean, std=std),
            ConvertDCHWtoCDHW()
        ])

        train_ds = CTImageDataset(chunked_csv=csv_loc,
                                  fold=fold,
                                  tr_test="train",
                                  transform=tr_aug if aug else tr_eval
                                  )
        val_ds = CTImageDataset(chunked_csv=csv_loc,
                                fold=fold,
                                tr_test="test",
                                transform=tr_eval
                                )

    return train_ds, val_ds
