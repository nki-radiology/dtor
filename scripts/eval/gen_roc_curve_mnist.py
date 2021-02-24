# -*- coding: utf-8 -*-
"""Process the outputs"""

__author__ = "Sean Benson"
__copyright__ = "MIT"

from dtor.utilities.utils_stats import roc_and_auc
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import torch
from dtor.utilities.utils import load_model, set_plt_config
from dtor.datasets.dataset_mnist import MNIST3DDataset
set_plt_config()

# Load test data
data = MNIST3DDataset(tr_test="test")
tot_folds = 1
prefix = "mnist"

# Process folds
res_preds = []
y_preds = []
y_labels = []
for f in range(tot_folds):
    # Get model for the fold
    model = load_model(prefix, f, 64)

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

# Ploting Receiving Operating Characteristic Curve
# Creating true and false positive rates
fp, tp, threshold1 = roc_curve(y_labels, y_preds)
#
auc, auc_cov, ci = roc_and_auc(y_preds, y_labels)
#
# Ploting ROC curves
plt.subplots(1, figsize=(10, 10))
plt.title('')
plt.plot(fp, tp, label=f'MNIST CNN, AUC={auc:.3f}')
plt.plot([0, 1], ls="--", color='gray')
plt.ylabel('Sensitivity')
plt.xlabel('1-Specificity')
plt.legend(loc='lower right')
plt.savefig(f"results/roc-{prefix}.png")
