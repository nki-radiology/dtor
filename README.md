DTOR
==============================

Torch based deep learning response prediction

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
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

Installation:
```shell script
git clone ...
cd dtor
conda create -p ./venv python=3.8 -y
conda activate ./venv
pip install -r requirements.txt
pip install -e .
```

Process dataset
```python
from dtor.dataset import CTImageDataset
dset = CTImageDataset("data/external/SPREADSHEET")
```
Note that this assumes folders in the external sub-folder conform to requirements

Run training
```shell script
export DTORROOT=$(pwd)
export CUDA_VISIBLE_DEVICES = "0,1" # for example
python dtor/trainer.py --datapoints PATH/TO/NEW/CSV --epochs N 
```

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
