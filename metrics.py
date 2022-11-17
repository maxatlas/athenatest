import torch
import matplotlib.pyplot as plt
import config as c
from pathlib import Path


def get_mce(ce, y_true):
    mce = max([ce_by_b[0] for ce_by_b in ce])
    return mce


def get_ece(ce, y_true):
    n = len(y_true)
    ece = sum([torch.mul(*ce_by_b) for ce_by_b in ce]) / n
    return ece


def get_ce(b_by_conf, y_conf_true: torch.Tensor, y_pred_true: torch.Tensor):
    """
    Bucketize y_conf_1d
    :param b_by_conf:
    :param y_conf_true: 1d confidence tensor for true labels
    :param y_pred_true: 1d binary prediction tensor for true labels
    :return: [(CE/bucket, n/bucket)]
    """
    n = len(y_conf_true)
    ce = [get_ce_per_bucket(b, b_by_conf, y_conf_true, y_pred_true) for b in range(c.n_class)]
    return ce


def get_ce_per_bucket(b: int, b_by_conf, y_conf_true, y_pred_true):
    indices_b = (b_by_conf == b).int()
    n = sum(indices_b)
    if n == 0:
        return 0, 1
    y_conf_b = torch.sum(torch.mul(y_conf_true, indices_b))
    y_pred_b = torch.sum(torch.mul(y_pred_true, indices_b))
    out = torch.abs(y_pred_b - y_conf_b) / n
    return out, torch.tensor(n)


def get_confusion_matrix(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return


def plot_confusion_matrix(cm: torch.Tensor, save_path: str, cmap=plt.cm.gray_r,
                          benchmark_session_id: str = ""):
    return


def plot_ce(ce: torch.Tensor, boundaries, save_file):
    ce = [float(ce_per_b[0]) for ce_per_b in ce]
    plt.bar(boundaries[1:]-1/(2*c.k), ce, width=1/c.k)
    plt.title("Calibrated Error over %i Bins" % c.k)
    plt.xticks(boundaries)
    plt.savefig(save_file)


def get_y_conf_true(y_conf_2d, y_true_1d):
    """

    :param y_conf_2d: shape (batch_size, n_class)
    :param y_true_1d: 1d true labels
    :return: 1d vec of confidence score for true labels only
    """
    batch_size = len(y_true_1d)
    out = y_conf_2d[torch.arange(batch_size), y_true_1d]
    return out


def get_y_pred_true(y_pred_1d, y_true_1d):
    """

    :param y_pred_1d: 1d vectors of length batch_size.
    :param y_true_1d: 1d true labels
    :return: 1d vec of predicted accuracy for true labels only
    """
    out = y_pred_1d.eq(y_true_1d).int().flatten()
    return out


