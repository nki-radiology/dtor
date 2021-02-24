import json
import matplotlib.pyplot as plt
from dtor.utilities.utils import set_plt_config
set_plt_config()

x = lambda p, t: f'results/res_{p}_totfolds_{t}.json'

files = [
    {'prefix': "tumor_nonzero_weighted_scaled_aug3_resnet", 'tot_folds': 3},
    {'prefix': "tumor_nonzero_weighted_scaled_aug5", 'tot_folds': 2},
    {'prefix': "tumor_nonzero_weighted_scaled", 'tot_folds': 3},
    {'prefix': "tumor_nonzero_weighted", 'tot_folds': 3},
    {'prefix': "tumor_nonzero", 'tot_folds': 3}
]
legname = {
    "tumor_nonzero_weighted_scaled_aug3_resnet": "Resnet (weighted, scaled+aug.)",
    "tumor_nonzero_weighted_scaled_aug5": "LTP CNN (weighted, scaled+aug.)",
    "tumor_nonzero_weighted_scaled": "LTP CNN (weighted, scaled)",
    "tumor_nonzero_weighted": "LTP CNN (weighted)",
    "tumor_nonzero": "LTP CNN"
}


# Ploting ROC curves
plt.subplots(1, figsize=(10, 10))
plt.title('')

for f in files:
    fname = x(f['prefix'], f['tot_folds'])
    with open(fname) as fout:
        res = json.load(fout)
    plt.plot(res['fp'], res['tp'], label=f'{legname[res["name"]]}, AUC={res["auc"]:.3f}')
plt.plot([0, 1], ls="--", color='gray')
plt.ylabel('Sensitivity')
plt.xlabel('1-Specificity')
plt.legend(loc='lower right')
plt.savefig(f"results/roc-combined.png")
