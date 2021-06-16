from dtor.logconf import logging
import numpy as np
from sklearn.metrics import f1_score
log = logging.getLogger(__name__)

# Used for computeBatchLoss and logMetrics to index into metrics_t/metrics_a
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3


def process_metrics(metrics_t, classification_threshold):
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
                    'correct/pos': pos_correct / np.float32(pos_count) * 100,
                    'pos_count': pos_count,
                    'neg_count': neg_count,
                    'pos_correct': pos_correct,
                    'neg_correct': neg_correct,
                    'neg_label_mask': neg_label_mask,
                    'pos_label_mask': pos_label_mask}
    return metrics_dict


def from_metrics_loss(rdict):
    return rdict[METRICS_LOSS_NDX].mean()


def from_metrics_f1(rdict):
    return f1_score(rdict[METRICS_LABEL_NDX].numpy(), rdict[METRICS_PRED_NDX].numpy())


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=3, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            log.info(f"Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                log.info('Early stopping')
                self.early_stop = True
