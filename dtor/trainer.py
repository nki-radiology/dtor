# -*- coding: utf-8 -*-
"""Master training script"""

__author__ = "Sean Benson"
__copyright__ = "MIT"

import datetime
import os
import sys
import random
import json
import time

import numpy as np
import pandas as pd
import pathlib

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dtor.utilities.utils import focal_loss
import torch.nn.functional as F

from dtor.loss.diceloss import DiceLoss
from dtor.logconf import enumerate_with_estimate
from dtor.logconf import logging
from dtor.utilities.utils import find_folds, get_class_weights
from dtor.utilities.model_retriever import model_choice
from dtor.utilities.data_retriever import get_data
from dtor.opts import init_parser
from dtor.opts import norms

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# Used for computeBatchLoss and logMetrics to index into metrics_t/metrics_a
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3


class TrainerBase:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        self.totalTrainingSamples_count = 0
        self.model = None
        self.weights = None
        self.trn_writer = None
        self.val_writer = None
        self.optimizer = None
        self.scheduler = None
        self.root_dir = os.environ["DTORROOT"]

        parser = init_parser()
        args = parser.parse_args(sys_argv)
        if args.load_json:
            with open(args.load_json, 'r') as f:
                args.__dict__ = json.load(f)
        self.cli_args = args
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # Needed to make training reproducible
        seed_value = self.cli_args.seed
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        random.seed(seed_value)
        if self.use_cuda: 
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def init_model(self, sample=None):
        return NotImplementedError

    def init_data(self, fold, mean=None, std=None):
        return NotImplementedError

    def init_optimizer(self):
        optim = Adam(self.model.parameters(), lr=self.cli_args.learnRate)
        decay = self.cli_args.decay
        scheduler = None
        if decay < 1.0:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=decay)
        return optim, scheduler

    def init_loaders(self, train_ds, val_ds):
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl, val_dl

    def init_tensorboard_writers(self, fold):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=f"{log_dir}-{fold}-trn_cls-{self.cli_args.comment}"
            )
            self.val_writer = SummaryWriter(
                log_dir=f"{log_dir}-{fold}-val_cls-{self.cli_args.comment}"
            )

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        # Load chunks file
        if self.cli_args.dset == "ltp":
            _df = pd.read_csv(self.cli_args.datapoints, sep="\t")
            tot_folds = find_folds(_df)
            log.info(f'Found a total of {tot_folds} folds to process')
        else:
            tot_folds = 1

        for fold in range(tot_folds):
            # Print
            log.info(f'FOLD {fold}')
            log.info('--------------------------------')

            # Data
            mean, std = norms[self.cli_args.norm]
            train_ds, val_ds, train_dl, val_dl = self.init_data(fold, mean=mean, std=std)

            # Get a sample batch
            sample = []
            for n, point in enumerate(train_dl):
                if n == 1:
                    break
                x = point[0]
                sample.append(x)
            sample = torch.cat(sample, dim=0)

            # Generate weights
            log.info('Calculating class weights')
            self.weights = get_class_weights(train_ds)
            self.weights = self.weights.to(self.device)

            # Model
            log.info('Initializing model')
            self.model = self.init_model(sample=sample)
            log.info('Model initialized')
            self.totalTrainingSamples_count = 0
            self.optimizer, self.scheduler = self.init_optimizer()
            log.info('Optimizer initialized')

            # If model is using cnn_finetune, we need to update the transform with the new
            # mean and std deviation values
            try:
                dpm = self.model if not self.use_cuda else self.model.module
            except nn.modules.module.ModuleAttributeError:
                dpm = self.model

            if hasattr(dpm, "original_model_info"):
                log.info('*******************USING PRETRAINED MODEL*********************')
                mean = dpm.original_model_info.mean
                std = dpm.original_model_info.std
                train_ds, val_ds, train_dl, val_dl = self.init_data(fold, mean=mean, std=std)

            log.info('*******************NORMALISATION DETAILS*********************')
            log.info(f"preprocessing mean: {mean}, std: {std}")

            for epoch_ndx in range(1, self.cli_args.epochs + 1):
                log.info("FOLD {}, Epoch {} of {}, {}/{} batches of size {}*{}".format(
                    fold,
                    epoch_ndx,
                    self.cli_args.epochs,
                    len(train_dl),
                    len(val_dl),
                    self.cli_args.batch_size,
                    (torch.cuda.device_count() if self.use_cuda else 1),
                ))

                trn_metrics_t = self.do_training(fold, epoch_ndx, train_dl)
                self.log_metrics(fold, epoch_ndx, 'trn', trn_metrics_t)

                val_metrics_t = self.do_validation(fold, epoch_ndx, val_dl)
                self.log_metrics(fold, epoch_ndx, 'val', val_metrics_t)

            model_path = os.path.join(pathlib.Path(os.environ["DTORROOT"]),
                                      f"results/model-{self.cli_args.tb_prefix}-fold{fold}.pth")
            torch.save(self.model.state_dict(), model_path)

            if hasattr(self, 'trn_writer'):
                self.trn_writer.close()
                self.val_writer.close()

            self.trn_writer = None
            self.val_writer = None

        # Save CLI args
        cli_name = f'results/model-{self.cli_args.tb_prefix}-{time.strftime("%Y%m%d-%H%M%S")}.json'
        with open(cli_name, 'w') as f:
            json.dump(self.cli_args.__dict__, f, indent=2)

    def do_training(self, fold, epoch_ndx, train_dl, num_augmentations=None):
        self.model = self.model.train().to(self.device)
        trn_metrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device
        )

        batch_iter = enumerate_with_estimate(
            train_dl,
            "F{}, E{} Training".format(fold, epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:

            if self.cli_args.augments > 0:
                for _ in range(self.cli_args.augments):
                    input_t, label_t, _ = batch_tup

                    self.optimizer.zero_grad()
                    loss_var = self.compute_batch_loss(
                        batch_ndx,
                        batch_tup,
                        train_dl.batch_size,
                        trn_metrics_g
                    )

                    loss_var.backward()
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()
            else:
                self.optimizer.zero_grad()
                loss_var = self.compute_batch_loss(
                    batch_ndx,
                    batch_tup,
                    train_dl.batch_size,
                    trn_metrics_g
                )

                loss_var.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trn_metrics_g.to('cpu')

    def do_validation(self, fold, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model = self.model.eval()
            val_metrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerate_with_estimate(
                val_dl,
                "F{} E{} Validation ".format(fold, epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.compute_batch_loss(
                    batch_ndx, batch_tup, val_dl.batch_size, val_metrics_g)

        return val_metrics_g.to('cpu')

    def compute_batch_loss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, _ = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        input_g=input_g.float()
        if self.cli_args.dim == 2:
            logits_g = self.model(input_g)
            probability_g = nn.Softmax(dim=1)(logits_g)
        else:
            logits_g, probability_g = self.model(input_g)

        CE = nn.CrossEntropyLoss(reduction='none', weight = self.weights)
        if "focal" in self.cli_args.loss.lower():
            loss_string = self.cli_args.loss
            parts = loss_string.split("_")
            assert len(parts) == 3, "Focal loss requires 'focal_ALPHA_BETA' formatting"
            alpha = float(parts[1])
            gamma = float(parts[2])
            loss_g = focal_loss(CE(logits_g, label_g), label_g, gamma, alpha)
        else:
            loss_g = CE(logits_g, label_g)
        
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)
        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g.detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = probability_g[:, 1].detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g.detach()

        return loss_g.mean()

    def log_metrics(
            self,
            fold,
            epoch_ndx,
            mode_str,
            metrics_t,
            classification_threshold=0.5,
    ):
        self.init_tensorboard_writers(fold)
        log.info("F{} E{} {}".format(
            fold,
            epoch_ndx,
            type(self).__name__,
        ))

        neg_label_mask = metrics_t[METRICS_LABEL_NDX] <= classification_threshold
        neg_pred_mask = metrics_t[METRICS_PRED_NDX] <= classification_threshold

        pos_label_mask = ~neg_label_mask
        pos_pred_mask = ~neg_pred_mask

        neg_count = int(neg_label_mask.sum())
        pos_count = int(pos_label_mask.sum())

        neg_correct = int((neg_label_mask & neg_pred_mask).sum())
        pos_correct = int((pos_label_mask & pos_pred_mask).sum())

        metrics_dict = {'loss/all': metrics_t[METRICS_LOSS_NDX].mean(),
                        'loss/neg': metrics_t[METRICS_LOSS_NDX, neg_label_mask].mean(),
                        'loss/pos': metrics_t[METRICS_LOSS_NDX, pos_label_mask].mean(),
                        'correct/all': (pos_correct + neg_correct) \
                                       / np.float32(metrics_t.shape[1]) * 100,
                        'correct/neg': neg_correct / np.float32(neg_count) * 100,
                        'correct/pos': pos_correct / np.float32(pos_count) * 100}

        log.info(
            ("F{} E{} {:8} {loss/all:.4f} loss, "
             + "{correct/all:-5.1f}% correct, "
             ).format(
                fold,
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )
        log.info(
            ("F{} E{} {:8} {loss/neg:.4f} loss, "
             + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
             ).format(
                fold,
                epoch_ndx,
                mode_str + '_neg',
                neg_correct=neg_correct,
                neg_count=neg_count,
                **metrics_dict,
            )
        )
        log.info(
            ("F{} E{} {:8} {loss/pos:.4f} loss, "
             + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
             ).format(
                fold,
                epoch_ndx,
                mode_str + '_pos',
                pos_correct=pos_correct,
                pos_count=pos_count,
                **metrics_dict,
            )
        )

        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

        writer.add_pr_curve(
            'pr',
            metrics_t[METRICS_LABEL_NDX],
            metrics_t[METRICS_PRED_NDX],
            self.totalTrainingSamples_count,
        )

        bins = [x / 50.0 for x in range(51)]

        neg_hist_mask = neg_label_mask & (metrics_t[METRICS_PRED_NDX] > 0.01)
        pos_hist_mask = pos_label_mask & (metrics_t[METRICS_PRED_NDX] < 0.99)

        if neg_hist_mask.any():
            writer.add_histogram(
                'is_neg',
                metrics_t[METRICS_PRED_NDX, neg_hist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )
        if pos_hist_mask.any():
            writer.add_histogram(
                'is_pos',
                metrics_t[METRICS_PRED_NDX, pos_hist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )


class Trainer(TrainerBase):
    def __init__(self):
        super().__init__()

    def init_model(self, sample=None):
        model = model_choice(self.cli_args.model, 
                resume=self.cli_args.resume, sample=sample,
                pretrain_loc=self.cli_args.pretrain_loc,
                pretrained_2d_name=self.cli_args.pretrained_2d_name)

        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def init_data(self, fold, mean=None, std=None):
        aug = False
        if self.cli_args.augments > 0:
            aug = True
        if mean:
            train_ds, val_ds = get_data(self.cli_args.dset, self.cli_args.datapoints, fold, aug=aug,
                                        mean=mean, std=std, dim=self.cli_args.dim)
        else:
            train_ds, val_ds = get_data(self.cli_args.dset, self.cli_args.datapoints, fold, aug=aug,
                    dim=self.cli_args.dim)
        train_dl, val_dl = self.init_loaders(train_ds, val_ds)
        return train_ds, val_ds, train_dl, val_dl


if __name__ == '__main__':
    Trainer().main()
