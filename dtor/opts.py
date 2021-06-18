# -*- coding: utf-8 -*-
"""Process all the options"""

__author__ = "Sean Benson"
__copyright__ = "MIT"

import json
import argparse


norms = {
    'torchvision_3d': ([0.43216, 0.394666, 0.37645],
                       [0.22803, 0.22145, 0.216989]),
    'torchvision_2d': ([0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225])
}


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-workers',
                        help='Number of worker processes for background data loading',
                        default=2,
                        type=int,
                        nargs='?',
                        )
    parser.add_argument('--batch-size',
                        help='Batch size to use for training',
                        default=32,
                        type=int,
                        nargs='?',
                        )
    parser.add_argument('--epochs',
                        help='Number of epochs to train for',
                        default=1,
                        type=int,
                        nargs='?',
                        )
    parser.add_argument('--seed',
                        help='Reproduce results with a certain seed',
                        default=42,
                        type=int,
                        nargs='?',
                        )
    parser.add_argument('--exp_name',
                        default='test',
                        nargs='?',
                        help="Experiment name, will be used for model saving and logs",
                        )
    parser.add_argument('--datapoints',
                        help="Location of the CSV with our points and labels",
                        type=str,
                        nargs='?',
                        default="data/chunked.csv",
                        )
    parser.add_argument('--loss',
                        help="What kind of loss function to use",
                        type=str,
                        nargs='?',
                        default="crossentropy",
                        )
    parser.add_argument('--model',
                        help="Which model do we want",
                        type=str,
                        nargs='?',
                        default="nominal",
                        )
    parser.add_argument('--load_json',
                        help="Specify args with a json",
                        type=str,
                        nargs='?',
                        default="",
                        )
    parser.add_argument('--best_json',
                        help="Specify args with best hyperparams",
                        type=str,
                        nargs='?',
                        default="",
                        )
    parser.add_argument('--dset',
                        help="Which dataset to use",
                        type=str,
                        nargs='?',
                        default="ltp",
                        )
    parser.add_argument('--augments',
                        help="How many rounds of augmentations",
                        type=int,
                        nargs='?',
                        default="0",
                        )
    parser.add_argument('--resume',
                        help="Checkpoint to resume from",
                        type=str,
                        nargs='?',
                        default="",
                        )
    parser.add_argument('--pretrain_loc',
                        help="Feature extraction pretrained location",
                        type=str,
                        nargs='?',
                        default="",
                        )
    parser.add_argument('--pretrained_2d_name',
                        help="Which 2d pretrained model to use",
                        type=str,
                        nargs='?',
                        default="",
                        )
    parser.add_argument('--dim',
                        help="Dimensions of input data",
                        type=int,
                        nargs='?',
                        default=3,
                        )
    parser.add_argument('--sam',
                        help="Do we want to turn on the SAM overfitting protection",
                        type=bool,
                        nargs='?',
                        default=False,
                        )
    parser.add_argument('--dset_lim',
                        help="How many data points to we want (MNIST OPTION ONLY)",
                        type=int,
                        nargs='?',
                        default=150,
                        )
    parser.add_argument('--earlystopping',
                        help="Early stopping to avoid overfitting, 0=off, n>0 is patience",
                        type=int,
                        nargs='?',
                        default=3,
                        )
    parser.add_argument('--num_trials',
                        help="How many trials for the tuning",
                        type=int,
                        nargs='?',
                        default=10,
                        )
    parser.add_argument('--mode',
                        help="train | tune",
                        type=str,
                        nargs='?',
                        default="train",
                        )
    parser.add_argument('--fix_nlayers',
                        help="How many layers of the input model do we fix",
                        type=int,
                        nargs='?',
                        default=0,
                        )
    parser.add_argument('--rn_depth',
                        help="How big a resnet do you want",
                        type=int,
                        nargs='?',
                        default=101,
                        )
    parser.add_argument('--rn_nclasses',
                        help="How many classes were in the original training",
                        type=int,
                        nargs='?',
                        default=700,
                        )
    return parser
