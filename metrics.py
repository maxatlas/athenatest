import torch
import matplotlib.pyplot as plt
import config as c

from utils import get_boundaries


def get_ece_mce_ce(y_pred_1d: torch.Tensor, y_conf_2d: torch.Tensor, y_true_1d: torch.Tensor, bin_size: int = c.k):
    """

    :param y_pred_1d:
    :param y_conf_2d:
    :param y_true_1d:
    :param bin_size:
    :return:
    """
    assert y_pred_1d.shape == y_true_1d.shape

    y_pred_true = get_y_pred_true(y_pred_1d, y_true_1d)
    y_conf_1d = get_y_conf_1d(y_conf_2d, y_pred_1d)
    batch_size = len(y_pred_true)

    boundaries = get_boundaries(bin_size)
    b_by_conf = torch.bucketize(y_conf_1d, boundaries) - 1
    ce = get_ce(b_by_conf, y_conf_1d, y_pred_true, bin_size=bin_size)

    mce = get_mce(ce)
    ece = get_ece(ce, batch_size=batch_size)

    return ece, mce, ce


def get_mce(ce):
    mce = max([ce_by_b[0] for ce_by_b in ce])
    return mce


def get_ece(ce, batch_size):
    ece = sum([torch.mul(*ce_by_b) for ce_by_b in ce]) / batch_size
    return ece


def get_ce(b_by_conf, y_conf_true: torch.Tensor, y_pred_true: torch.Tensor, bin_size: int = c.k):
    """

    :param b_by_conf: list of bucket indices per confidence score.
    :param y_conf_true: 1d vec of confidence scores for true labels only
    :param y_pred_true: 1d vec of binary prediction for true labels only
    :param bin_size: bin size
    :return: [(CE per bucket, size of bucket)]
    """
    ce = [get_ce_per_bucket(b, b_by_conf, y_conf_true, y_pred_true) for b in range(bin_size)]
    return ce


def get_ce_per_bucket(b: int, b_by_conf, y_conf_1d, y_pred_true):
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
    assert y_true.shape == y_pred.shape

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


def get_y_conf_1d(y_conf_2d, y_pred_1d):
    """

    :param y_conf_2d: shape (actual_batch_size, n_class)
    :param y_pred_1d: 1d predicted labels
    :return: 1d vec of confidence score for predicted labels only
    """
    batch_size = len(y_pred_1d)
    out = y_conf_2d[torch.arange(batch_size), y_pred_1d]
    return out


def get_y_pred_true(y_pred_1d, y_true_1d):
    """

    :param y_pred_1d: 1d vectors of length batch_size.
    :param y_true_1d: 1d true labels
    :return: 1d vec of predicted accuracy for true labels only
    """
    out = y_pred_1d.eq(y_true_1d).int().flatten()
    return out


def plot_ce(ce: torch.Tensor, save_path, bin_size=c.k):
    boundaries = get_boundaries(bin_size)
    ce = [float(ce_per_b[0]) for ce_per_b in ce]
    plt.bar(boundaries[1:]-1/(2*bin_size), ce, width=1/bin_size)
    plt.title("Calibrated Error over %i Bins" % bin_size)
    plt.xticks(boundaries)
    plt.savefig(save_path)

    plt.clf()


def plot_confusion_matrix(cm: torch.Tensor, save_path: str, cmap=plt.cm.gray_r,
                          benchmark_session_id: str = ""):
    dim_cm = cm.shape[0]
    plt.matshow(cm, cmap=cmap)
    plt.title("Confusion Matrix\nfor %s" % benchmark_session_id)
    plt.colorbar()
    tick_marks = range(dim_cm)
    plt.xticks(tick_marks, range(dim_cm))
    plt.yticks(tick_marks, range(dim_cm))
    plt.ylabel("True class")
    plt.xlabel("Predicted class")
    plt.savefig(save_path)

    plt.clf()

