# -*- coding: utf-8 -*-
"""Process the outputs and generate ROC"""

__author__ = "Sean Benson"
__copyright__ = "MIT"

from dtor.utilities.utils_stats import stats_from_results
import numpy as np
import argparse
from dtor.utilities.utils import set_plt_config
from dtor.utilities.model_retriever import load_model
from dtor.datasets.dataset_nominal import CTImageDataset
set_plt_config()

parser = argparse.ArgumentParser()
parser.add_argument("--tot_folds", type=int, help="Number of folds for model training",
                    default=3)
parser.add_argument("--prefix", type=str, help="Training prefix",
                    default="default")
parser.add_argument("--mode", type=str, help="Data combination mode",
                    default="concat")
parser.add_argument("--legname", type=str, help="Legend description",
                    default='LTP CNN')
args = parser.parse_args()
tot_folds = args.tot_folds
prefix = args.prefix
mode = args.mode
legname = args.legname

# Process folds
res_preds = []
for f in range(tot_folds):
    # Load test data
    data = CTImageDataset(fold=f, tr_test="test", chunked_csv="data/chunked.csv")
    
    # Make sample for loading
    _f, _, _ = data[0]
    sample = _f.unsqueeze(0)
    
    # Get model for the fold
    model = load_model(prefix, f, "nominal", sample=sample)

    # Generate vector of predictions and true labels
    y_preds = dict()

    for n in range(len(data)):
        f, truth, extra = data[n]

        key = '_'.join(extra[1].split('_')[:2])
        weight = extra[0]
        true_key = f"{key}_truth"
        weights_key = f"{key}_weights"
        if key not in list(y_preds.keys()):
            y_preds[key] = []
            y_preds[weights_key] = []

        x = f.unsqueeze(0)
        l, p = model(x)

        pred = p[0][1].detach().numpy()

        y_preds[key].append(pred)
        y_preds[weights_key].append(weight)
        y_preds[true_key] = truth
    res_preds.append(y_preds)

# Concatenate results of the folds
y_preds_total = []
y_labels_total = []
if mode == 'concat':
    for r in res_preds:
        abls = list(r.keys())
        abls = ['_'.join(k.split("_")[:2]) for k in abls]
        abls = list(set(abls))
        for k in abls:
            y_preds_total.append(r[k][np.argmax(r[f"{k}_weights"])])
            y_labels_total.append(r[f"{k}_truth"])
else:
    raise NotImplementedError

y_labels_total = np.array(y_labels_total)
y_preds_total = np.array(y_preds_total)

output_name = f"results/roc-{prefix}.png"
res_name = f"results/res-{prefix}.json"
stats_from_results(y_preds_total, y_labels_total, results_name=res_name, plot_name=output_name, legname=legname)
