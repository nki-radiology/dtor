# -*- coding: utf-8 -*-
"""MNIST dataset for testing"""

__author__ = "Sean Benson"
__copyright__ = "MIT"

import torch
from torch.utils.data import Dataset
import h5py
import matplotlib.pyplot as plt
import numpy as np

def array_to_color(array, cmap="Oranges"):
    s_m = plt.cm.ScalarMappable(cmap=cmap)
    return s_m.to_rgba(array)[:,:-1]

def data_transform(data, labels, req=[0, 1]):
    data_t = []
    label_t = []
    for i in range(data.shape[0]):
        if labels[i] not in req:
            #print(labels[i])
            continue
        i_data = array_to_color(data[i]).reshape(16, 16, 16, 3)
        i_data = np.moveaxis(i_data, -1, 0) 
        data_t.append(i_data)
        label_t.append(labels[i])
    return np.asarray(data_t, dtype=np.float32), np.asarray(label_t)

class MNIST3DDataset(Dataset):
    """Dataset from 3D MNIST point cloud for testing"""

    def __init__(self, tr_test=None):
        with h5py.File("/Users/sbenson/OpenData/mnist3d/full_dataset_vectors.h5", "r") as hf:    
            datapoints = list(hf.keys())
            print(f"Found {list(hf.keys())} keys")
            self.X = hf[f"X_{tr_test}"][:]
            self.y = hf[f"y_{tr_test}"][:]

        self.X, self.y = data_transform(self.X, self.y) 
        print(f"Kept {self.y.shape} data points")

    def __len__(self):
        return self.y.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return [self.X[idx], self.y[idx], 0]
