# -*- coding: utf-8 -*-
"""MNIST dataset for testing"""

__author__ = "Sean Benson"
__copyright__ = "MIT"

import torch
from torch.utils.data import Dataset
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample_poly

_factors = [(3, 1), (3, 1), (3, 1)]

def array_to_color(array, cmap="Oranges"):
    s_m = plt.cm.ScalarMappable(cmap=cmap)
    return s_m.to_rgba(array)[:, :-1]


def data_transform(data, labels, req=[0, 1]):
    data_t = []
    label_t = []
    for i in range(data.shape[0]):
        if labels[i] not in req:
            continue
        i_data = array_to_color(data[i]).reshape(16, 16, 16, 3)
        for k in range(3):
            i_data = resample_poly(i_data, _factors[k][0], _factors[k][1], axis=k)
        i_data = np.moveaxis(i_data, -1, 0)
        data_t.append(i_data)
        label_t.append(labels[i])
    return np.asarray(data_t, dtype=np.float32), np.asarray(label_t)


class MNIST3DDataset(Dataset):
    """
    Dataset from 3D MNIST point cloud for testing
    """

    def __init__(self, h5_file, tr_test=None, transform=None, limit=None):
        """
        Initialization

        :param h5_file: data location
        :param tr_test: train or test subset
        :param transform: transform
        """

        with h5py.File(h5_file, "r") as hf:
            datapoints = list(hf.keys())
            print(f"Found {list(hf.keys())} keys")
            if limit:
                self.X = hf[f"X_{tr_test}"][:limit]
                self.y = hf[f"y_{tr_test}"][:limit]
            else:
                self.X = hf[f"X_{tr_test}"][:]
                self.y = hf[f"y_{tr_test}"][:]

        self.X, self.y = data_transform(self.X, self.y) 
        print(f"Kept {self.y.shape} data points")

        self.transform = transform

    def __len__(self):
        return self.y.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.X[idx]
        image = torch.from_numpy(image).to(torch.float32)
        if self.transform:
            image = self.transform(image)

        return [image, self.y[idx], 0]
