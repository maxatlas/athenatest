import torch
import matplotlib.pyplot as plt
import config as c
from pathlib import Path
from utils import get_boundaries


def get_mce(ce_b):
    mce = max([torch.abs(ce_by_b[0]) for ce_by_b in ce_b])
    return mce


def get_ece(ce, batch_size: int):
    # CE per bucket * size of bucket
    # sum the absolute of product
    ece = sum(ce) / batch_size
    return ece


def get_ce(ce_b, absolute=False):
    """

    :param ce_b: torch.tensor(([+/-CE/bucket, size of bucket]))
    :param absolute: return the absolute tensor if true, for ece/mce calculation;
                     change nothing if false, for aggregation.
    :return: torch.tensor([un-absolute CE per bucket * size of bucket])
    """
    ce = torch.tensor([torch.mul(*pair) for pair in ce_b])
    ce = torch.abs(ce) if absolute else ce
    return ce


def get_ce_b(y_pred_1d: torch.Tensor, y_conf_2d: torch.Tensor, y_true_1d: torch.Tensor, bin_size: int = c.k) -> \
        torch.Tensor:
    """

    :param y_pred_1d: 1d vec of predicted labels
    :param y_conf_2d: 2d vec of confidence score of shape (batch_size, n_class)
    :param y_true_1d: 1d vec of true labels
    :param bin_size: used to generate boundaries.
    :return: torch.tensor([(un-absolute CE per bucket, size of bucket)])
    """
    assert y_pred_1d.dim() == 1
    assert y_conf_2d.dim() == 2
    assert y_true_1d.dim() == 1
    assert y_pred_1d.shape == y_true_1d.shape
    assert len(y_conf_2d) == len(y_pred_1d)

    y_pred_true = get_y_pred_true(y_pred_1d, y_true_1d)
    y_conf_1d = get_y_conf_1d(y_conf_2d, y_pred_1d)

    boundaries = get_boundaries(bin_size)
    b_by_conf = torch.bucketize(y_conf_1d, boundaries) - 1

    ce_b = [get_ce_per_bucket(b, b_by_conf, y_conf_1d, y_pred_true) for b in range(bin_size)]
    ce_b = torch.tensor(ce_b)

    return ce_b


def get_ce_per_bucket(b: int, b_by_conf, y_conf_1d, y_pred_true) -> tuple[torch.Tensor, torch.Tensor]:
    """

    :param b: current bucket to evaluate
    :param b_by_conf: list of bucket indices per confidence score.
    :param y_conf_1d: 1d vec of confidence scores for predicted labels only
    :param y_pred_true: 1d vec of binary prediction for true labels only
    :return: (un-absolute CE per bucket, size of bucket)
    """
    indices_b = (b_by_conf == b).int()
    n = sum(indices_b)
    if n == 0:
        return torch.tensor(0), torch.tensor(1)
    # sum of predicted confidence under bucket b
    y_conf_b = torch.sum(torch.mul(y_conf_1d, indices_b))
    # number of correctly predicted samples under bucket b
    y_pred_b = torch.sum(torch.mul(y_pred_true, indices_b))
    out = (y_pred_b - y_conf_b) / n
    return out, n


def get_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, n_class: int, cm_last: torch.Tensor = None) -> \
        torch.Tensor:
    """

    :param y_true: 1d vec of true labels
    :param y_pred: 1d vec of predicted labels
    :param n_class: number of classes. Default on value specified in config.py
    :param cm_last: calculation will aggregate upon cm_last. Will initiate to empty matrix if not given.
    :return: confusion matrix of shape (n_class, n_class)
    """
    assert y_true.shape == y_pred.shape
    assert y_true.dim() == 1
    assert y_pred.dim() == 1

    if cm_last is None: cm_last = init_confusion_matrix(n_class)
    y_true = y_true.int()
    y_pred = y_pred.int()
    cm = cm_last.clone().detach().requires_grad_(False)
    for true_i, pred_i in zip(y_true, y_pred):
        cm[pred_i][true_i] += 1
    return cm


def init_confusion_matrix(n_class: int = c.n_class) -> torch.Tensor:
    cm = torch.zeros(n_class, n_class).int()
    return cm


def get_y_conf_1d(y_conf_2d: torch.Tensor, y_pred_1d: torch.Tensor) -> torch.Tensor:
    """

    :param y_conf_2d: shape (actual_batch_size, n_class)
    :param y_pred_1d: 1d predicted labels
    :return: 1d vec of confidence score for predicted labels only
    """
    assert y_conf_2d.dim() == 2
    assert y_pred_1d.dim() == 1
    assert len(y_conf_2d) == len(y_pred_1d)

    batch_size = len(y_pred_1d)
    out = y_conf_2d[torch.arange(batch_size), y_pred_1d]
    return out


def get_y_pred_true(y_pred_1d: torch.Tensor, y_true_1d: torch.Tensor):
    """

    :param y_pred_1d: 1d vectors of length batch_size.
    :param y_true_1d: 1d true labels
    :return: 1d vec of predicted accuracy for true labels only
    """
    assert y_pred_1d.dim() == 1
    assert y_true_1d.dim() == 1
    assert y_pred_1d.shape == y_true_1d.shape

    out = y_pred_1d.eq(y_true_1d).int().flatten()
    return out


def plot_ce(ce, save_path: Path, bin_size: int = c.k, batch_size: int = c.batch_size_test):
    assert len(ce) == bin_size
    ce = torch.abs(ce) / batch_size

    boundaries = get_boundaries(bin_size)
    ce = [float(ce_per_b) for ce_per_b in ce]
    plt.bar(boundaries[1:]-1/(2*bin_size), ce, width=1/bin_size)
    plt.title("Calibration Error over %i Bins" % bin_size)
    plt.xticks(boundaries)
    plt.savefig(str(save_path))

    plt.clf()


def plot_confusion_matrix(cm: torch.Tensor, save_path: Path, cmap=plt.cm.gray_r):
    dim_cm = cm.shape[0]
    plt.matshow(cm, cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = range(dim_cm)
    plt.xticks(tick_marks, range(dim_cm))
    plt.yticks(tick_marks, range(dim_cm))
    plt.ylabel("True class")
    plt.xlabel("Predicted class")
    plt.savefig(str(save_path))

    plt.clf()

