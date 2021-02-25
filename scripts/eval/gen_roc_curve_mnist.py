# -*- coding: utf-8 -*-
"""Process the outputs"""

__author__ = "Sean Benson"
__copyright__ = "MIT"

from dtor.utilities.utils_stats import stats_from_results
import numpy as np
import torch
from dtor.utilities.utils import load_model, set_plt_config
from dtor.datasets.dataset_mnist import MNIST3DDataset
set_plt_config()

# Load test data
data = MNIST3DDataset(h5_file="data/external/mnist/full_dataset_vectors.h5", tr_test="test")
tot_folds = 1
prefix = "mnist_example"

# Process folds
res_preds = []
y_preds = []
y_labels = []
for f in range(tot_folds):
    # Get model for the fold
    model = load_model(prefix, f, model_type="nominal_mnist")

    # Generate vector of predictions and true labels
    y_preds = []
    y_labels = []

    for n in range(400):
        f, truth, _ = data[n]
        x = torch.from_numpy(f).unsqueeze(0)
        l, p = model(x)

        pred = p[0][1].detach().numpy()

        y_preds.append(pred)
        y_labels.append(truth)
    res_preds.append(y_preds)
np_res = np.array(res_preds)
y_preds = np.mean(np_res, axis=0)
y_labels = np.array(y_labels)

output_name = f"results/roc-{prefix}.png"
stats_from_results(y_preds, y_labels, plot_name=output_name, legname="MNIST CNN")
