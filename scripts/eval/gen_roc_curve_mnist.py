# -*- coding: utf-8 -*-
"""Process the outputs"""

__author__ = "Sean Benson"
__copyright__ = "MIT"

from dtor.utilities.utils_stats import stats_from_results
import numpy as np
import torch
from dtor.utilities.utils import set_plt_config
from dtor.utilities.model_retriever import load_model
from dtor.datasets.dataset_mnist import MNIST3DDataset
set_plt_config()

# Load test data
data = MNIST3DDataset(h5_file="data/external/mnist/full_dataset_vectors.h5", tr_test="test")
tot_folds = 3
prefix = "default"

# Process folds
res_preds = []
y_preds = []
y_labels = []
for f in range(tot_folds):
    # Get model for the fold
    _f, _, _ = data[0]
    sample = _f.unsqueeze(0)
    use_cuda = torch.cuda.is_available()   
    device = torch.device("cuda" if use_cuda else "cpu")
    sample = sample.to(device)
    full_name=os.path.join("/home/marjaneh/ltp-prediction/results",  'model-' + str(prefix) +'-fold'+ str(f) +'.pth') 
    model = load_model(prefix, f, model_type="resnet18+dense",full_name=full_name, sample=sample)
    model = model.to(device)
    # Generate vector of predictions and true labels
    y_preds = []
    y_labels = []

    for n in range(400):
        f, truth, _ = data[n]
        x = f.unsqueeze(0)
        x = x.to(device)
        l, p = model(x)
        pred = p[0][1].detach().cpu()  #.numpy()

        y_preds.append(pred)
        y_labels.append(truth)
    res_preds.append(y_preds)
np_res = np.array(res_preds)
y_preds = np.mean(np_res, axis=0)
y_labels = np.array(y_labels)

output_name = f"results/roc-{prefix}.png"
stats_from_results(y_preds, y_labels, plot_name=output_name, legname="MNIST CNN")
