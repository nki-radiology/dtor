# -*- coding: utf-8 -*-
"""Process all the options"""

__author__ = "Sean Benson"
__copyright__ = "MIT"

import json
import argparse


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
    parser.add_argument('--tb-prefix',
                        default='test',
                        nargs='?',
                        help="Data prefix to use for Tensorboard run. Defaults to chapter.",
                        )
    parser.add_argument('--comment',
                        help="Comment suffix for Tensorboard run.",
                        nargs='?',
                        default='dwlpt',
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
    return parser


class ResNetOptions:
    def __init__(self, injson):
        injson = open(injson,"r")
        injson =json.load(injson)
        #
        self.model = injson['model']
        self.input_W = injson['input_W']
        self.input_D = injson['input_D']
        self.input_H = injson['input_H']
        self.model_depth = injson['model_depth']
        self.resnet_shortcut = injson['resnet_shortcut']
        self.n_seg_classes = injson['n_seg_classes']
        self.no_cuda=True if injson['no_cuda'] else False
        self.phase = injson['phase']
        self.pretrain_path = injson['pretrain_path']
        self.new_layer_names = injson['new_layer_names']
