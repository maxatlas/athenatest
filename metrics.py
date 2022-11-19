import torch
import matplotlib.pyplot as plt
import config as c

from utils import get_boundaries


def get_ece_mce_ce(y_pred_1d: torch.Tensor, y_conf_2d: torch.Tensor, y_true_1d: torch.Tensor, bin_size: int = c.k) -> \
        tuple[float, float, list]:
    """

    :param y_pred_1d: 1d vec of predicted labels
    :param y_conf_2d: 2d vec of confidence score of shape (batch_size, n_class)
    :param y_true_1d: 1d vec of true labels
    :param bin_size: used to generate boundaries.
    :return:
    """
    assert y_pred_1d.dim() == 1
    assert y_conf_2d.dim() == 2
    assert y_true_1d.dim() == 1
    assert y_pred_1d.shape == y_true_1d.shape
    assert len(y_conf_2d) == len(y_pred_1d)

    y_pred_true = get_y_pred_true(y_pred_1d, y_true_1d)
    y_conf_1d = get_y_conf_1d(y_conf_2d, y_pred_1d)
    batch_size = len(y_pred_true)

    boundaries = get_boundaries(bin_size)
    b_by_conf = torch.bucketize(y_conf_1d, boundaries) - 1
    ce = get_ce(b_by_conf, y_conf_1d, y_pred_true, bin_size=bin_size)

    mce = get_mce(ce)
    ece = get_ece(ce, batch_size=batch_size)

    ece, mce = round(float(ece), c.metrics_round_to),\
               round(float(mce), c.metrics_round_to),

    return ece, mce, ce


def get_mce(ce):
    mce = max([ce_by_b[0] for ce_by_b in ce])
    return mce


def get_ece(ce, batch_size: int):
    ece = sum([torch.mul(*ce_by_b) for ce_by_b in ce]) / batch_size
    return ece


def get_ce(b_by_conf, y_conf_true: torch.Tensor, y_pred_true: torch.Tensor, bin_size: int = c.k) -> list:
    """

    :param b_by_conf: list of bucket indices per confidence score.
    :param y_conf_true: 1d vec of confidence scores for true labels only
    :param y_pred_true: 1d vec of binary prediction for true labels only
    :param bin_size: bin size
    :return: [(CE per bucket, size of bucket)]
    """
    ce = [get_ce_per_bucket(b, b_by_conf, y_conf_true, y_pred_true) for b in range(bin_size)]
    return ce


def get_ce_per_bucket(b: int, b_by_conf, y_conf_1d, y_pred_true) -> tuple[torch.Tensor, torch.Tensor]:
    """

    :param b: current bucket to evaluate
    :param b_by_conf: list of bucket indices per confidence score.
    :param y_conf_1d: 1d vec of confidence scores for predicted labels only
    :param y_pred_true: 1d vec of binary prediction for true labels only
    :return: (CE per bucket, size of bucket)
    """
    indices_b = (b_by_conf == b).int()
    n = sum(indices_b)
    if n == 0:
        return torch.tensor(0), torch.tensor(1)
    # sum of predicted confidence under bucket b
    y_conf_b = torch.sum(torch.mul(y_conf_1d, indices_b))
    # number of correctly predicted samples under bucket b
    y_pred_b = torch.sum(torch.mul(y_pred_true, indices_b))
    out = torch.abs(y_pred_b - y_conf_b) / n
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

    if not cm_last: cm_last = init_confusion_matrix(n_class)
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


def plot_ce(ce, save_path: str, bin_size: int = c.k):
    assert len(ce) == bin_size

    boundaries = get_boundaries(bin_size)
    ce = [float(ce_per_b[0]) for ce_per_b in ce]
    plt.bar(boundaries[1:]-1/(2*bin_size), ce, width=1/bin_size)
    plt.title("Calibrated Error over %i Bins" % bin_size)
    plt.xticks(boundaries)
    plt.savefig(save_path)

    plt.clf()


def plot_confusion_matrix(cm: torch.Tensor, save_path: str, cmap=plt.cm.gray_r):
    dim_cm = cm.shape[0]
    plt.matshow(cm, cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = range(dim_cm)
    plt.xticks(tick_marks, range(dim_cm))
    plt.yticks(tick_marks, range(dim_cm))
    plt.ylabel("True class")
    plt.xlabel("Predicted class")
    plt.savefig(save_path)

    plt.clf()

