import torch
import matplotlib.pyplot as plt
import config as c
from pathlib import Path
from utils import get_boundaries


def get_mce(ce: torch.Tensor) -> torch.Tensor:
    """
    Calculate MCE
    :param ce:
    :return:
    """
    mce = torch.max(torch.abs(ce).nan_to_num(-0.1))
    return mce


def get_ece(ce_b, batch_size: int):
    # CE per bucket * size of bucket
    # sum the absolute of product

    # if CE is the initialized one (all zeros), return negative value to indicate nan.
    if all(ce_b[:, 0]) == 0:
        return torch.tensor(-0.1)
    ece = sum(ce_b[:, 0].nan_to_num(0).abs()) / batch_size
    return ece


def get_ce(ce_b):
    """

    :param ce_b: torch.tensor([+/-CE per bucket, size of bucket])
    :return: torch.tensor([CE per bucket * size of bucket])
    """
    ce = torch.tensor([torch.div(*pair) for pair in ce_b]).abs()
    return ce


def get_ce_b(y_pred_1d: torch.Tensor, y_conf_2d: torch.Tensor, y_true_1d: torch.Tensor,
             bin_size: torch.Tensor = torch.tensor(c.k)) -> torch.Tensor:
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
    bucket_size = sum(indices_b)
    if bucket_size == 0:
        return torch.tensor(torch.nan), torch.tensor(1)
    # sum of predicted confidence under bucket b
    y_conf_b = torch.sum(torch.mul(y_conf_1d, indices_b))
    # number of correctly predicted samples under bucket b
    y_pred_b = torch.sum(torch.mul(y_pred_true, indices_b))
    ce_per_b = (y_pred_b - y_conf_b)
    return ce_per_b, bucket_size


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

    if cm_last is None:
        cm_last = init_confusion_matrix(n_class)
    y_true = y_true.int()
    y_pred = y_pred.int()
    cm = cm_last.clone().detach().requires_grad_(False)
    for true_i, pred_i in zip(y_true, y_pred):
        cm[pred_i][true_i] += 1
    return cm


def init_confusion_matrix(n_class: int) -> torch.Tensor:
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


def get_y_pred_true(y_pred_1d: torch.Tensor, y_true_1d: torch.Tensor) -> torch.Tensor:
    """

    :param y_pred_1d: 1d vectors of length batch_size.
    :param y_true_1d: 1d true labels
    :return: 1d vec of predicted accuracy for true labels only
    """
    assert y_pred_1d.shape == y_true_1d.shape
    assert y_true_1d.dim() == 1

    out = y_pred_1d.eq(y_true_1d).int().flatten()
    return out


def get_accuracy(y_pred_1d: torch.Tensor, y_true_1d: torch.Tensor) -> torch.Tensor:
    assert y_pred_1d.shape == y_true_1d.shape
    assert y_pred_1d.dim() == 1

    acc = sum(y_pred_1d.eq(y_true_1d))/len(y_pred_1d)
    return acc


def plot_ce(ce, save_path: Path, bin_size: torch.Tensor = torch.tensor(c.k), batch_size: int = c.batch_size_test):
    assert len(ce) == bin_size
    ce = (ce / batch_size).nan_to_num(-0.1)

    bin_size = bin_size.to("cpu")
    boundaries = get_boundaries(bin_size)
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
