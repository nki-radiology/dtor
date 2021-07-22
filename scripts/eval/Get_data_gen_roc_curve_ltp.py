# -*- coding: utf-8 -*-
"""Process the outputs and generate ROC"""

__author__ = "Sean Benson"
__copyright__ = "MIT"

from dtor.utilities.utils_stats import stats_from_results
import numpy as np
import argparse
import torch
from dtor.utilities.utils import set_plt_config
from dtor.utilities.model_retriever import load_model
from dtor.datasets.dataset_nominal import CTImageDataset
from dtor.datasets.dataset_mnist import MNIST3DDataset
set_plt_config()
import os
import torch.nn as nn
from dtor.utilities.data_retriever import get_data ##add new
from dtor.trainer import TrainerBase
import sys

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
#%%
sys.argv.extend(["--load_json", "/home/marjaneh/ltp-prediction/results/test-train/options.json"])


#%%
# Process folds
# Concatenate results of the folds
y_preds_total = []
y_labels_total = []
for f in range(tot_fold):
    # Load test data
   # data = CTImageDataset(fold=f, tr_test="test", chunked_csv="/home/marjaneh/ltp-prediction/data/chunked.csv", dim=2)
    train_ds, val_ds = get_data("ltp", "/home/marjaneh/ltp-prediction/data/chunked.csv", f,
                                        dim=2, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    A = TrainerBase()
  #  train_ds, val_ds, train_dl, val_dl = A.init_data(f, mean=[0.5,0.5], std=[0.5,0.5])
    train_dl, val_dl = A.init_loaders(train_ds,val_ds)
     
    #  Make sample for loading
    sample = []
    for n, point in enumerate(val_dl):
        if n == 1:
           break
        x = point[0]
        sample.append(x)
    sample = torch.cat(sample, dim=0)          

    use_cuda = torch.cuda.is_available()   #add new
    device = torch.device("cuda" if use_cuda else "cpu")     
    sample = sample.to(device) # till here
    full_name=os.path.join("/home/marjaneh/ltp-prediction/results/test-train/",  'model-' + 'test' +'-fold'+ str(f) +'.pth')  #str(prefix)
     #Get Model for fold
    
    model = load_model(prefix, f, "pretrained_2d",full_name=full_name, sample=None)
    model = model.to(device)        
    # Generate vector of predictions and true labels
    y_preds = dict()

    for n in range(len(val_ds)):
        f, truth, extra = val_ds[n]

        x = f.unsqueeze(0)
        x = x.to(device)
      #  l,p=model(x) #for 3d
        l = model(x)  #for 2d
        p = nn.Softmax(dim=1)(l)  #for 2d

        pred = p[0][1].detach().cpu()  #.numpy()
        y_preds_total.append(pred)
        y_labels_total.append(truth)

y_labels_total = np.array(y_labels_total)
y_preds_total = np.array(y_preds_total)

print(y_preds_total)
print(y_labels_total)
output_name = os.path.join("/home/marjaneh/ltp-prediction/results", '_train-roc-' + str(prefix) +'.png')
res_name = os.path.join("/home/marjaneh/ltp-prediction/results", '_train-res-' + str(prefix) +'.json')
stats_from_results(y_preds_total, y_labels_total, results_name=res_name, plot_name=output_name, legname=legname)
#%%
