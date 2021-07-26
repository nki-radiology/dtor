[![DOI](https://zenodo.org/badge/341508464.svg)](https://zenodo.org/badge/latestdoi/341508464)

DTOR
==============================

DTOR is a project originally designed to assist with the AI-based extraction of features
for treatment response prediction

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── config             <- JSON files for training configuration
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── cache          <- For quicker training
    │
    ├── models             <- Model source files
    │
    ├── dtor               <- Classes and utilities used that will be installed
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    |
    ├── results            <- Outputs from training and processing (model pth files, ROCs, AUCs, etc.)
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── scripts                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │
    │   ├── eval           <- Scripts to generate a ROC/AUC
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

Usage
-----

Installation and setup:
```shell script
git clone ...
cd dtor
conda create -p ./venv python=3.8 -y
conda activate ./venv
pip install -r requirements.txt
pip install -e .
export DTORROOT=$(pwd)
export CUDA_VISIBLE_DEVICES = "0,1" # for example, without this will either use all GPUs or CPU
```

Run training for MNIST3D example (data taken from [Kaggle](https://www.kaggle.com/daavoo/3d-mnist))
```shell script
python dtor/trainer.py --load_json config/default.json 
```

Process results:
```shell script
python scripts/eval/gen_roc_curve_mnist.py
```

Using your own dataset
-----

Datasets and Models are configurable but require the modification of `(data|mode)_retriever.py` utilities.

In such cases it is often best to create your own class that inherits from TrainerBase.
In this way you can train with your own slimmed-down package and add dtor to its `requirements.txt`:
```shell script
echo "git+ssh://git@github.com/tevien/dtor.git" >> requirements.txt
```

An example of subsequent usage:
```python
from dtor.trainer import TrainerBase
from WHEREVER_MY_MODIFIED_FUNCIONS_ARE import get_data, model_choice
import torch
import torch.nn as nn
import sys
if len(sys.argv) == 1:
    print("Usage:")
    print("python train.py --load_json PATH/TO/JSON")


# Initialise will take json config
class LTPTrainer(TrainerBase):
    def __init__(self):
        super().__init__()

    def init_model(self):
        model = model_choice(self.cli_args.model)
        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def init_data(self, fold):
        aug = False
        if self.cli_args.augments > 0:
            aug = True
        train_ds, val_ds = get_data(self.cli_args.datapoints, fold, aug=aug)
        train_dl, val_dl = self.init_loaders(train_ds, val_ds)
        return train_ds, val_ds, train_dl, val_dl
    
    def init_tune(self, trial):
        self.t_learnRate = trial.suggest_loguniform('learnRate', 1e-6, 1e-3)
        self.t_decay = trial.suggest_uniform('decay', 0.9, 0.99)
        self.t_alpha = trial.suggest_uniform('focal_alpha', 0.5, 3.0)
        self.t_gamma = trial.suggest_uniform('focal_gamma', 0.5, 5.0)
        self.patience = trial.suggest_int('earlystopping', 3, 6)
        if self.fix_nlayers:
            self.fix_nlayers = trial.suggest_int('fix_nlayers', 3, 6)


LTPTrainer().main()
```

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
