import torch
import matplotlib.pyplot as plt
import config as c
from pathlib import Path


def get_mce(y_conf: torch.Tensor, y_true: torch.Tensor):
    mce = 0
    return mce


def get_ece(y_conf_1d: torch.Tensor, y_pred_true: torch.Tensor):
    """
    Bucketize y_conf_1d
    :param y_conf_1d: 1d confidence tensor for true labels
    :param y_pred_true: 1d binary prediction tensor for true labels
    :return: ECE
    """
    boundaries = torch.arange(0, 1.01, 0.1)
    y_conf_by_b = torch.bucketize(y_conf_1d, boundaries)
    n = len(y_conf_1d)
    ece = sum([get_ece_per_bucket(b, y_conf_by_b, y_pred_true) for b in range(c.n_class)]) / n
    return ece


def get_ece_per_bucket(b:int, y_conf_by_b, y_pred_true):
    indices_b = (y_conf_by_b == b).nonzero()
    n = sum(indices_b)
    y_conf_b = torch.sum(y_conf_by_b[indices_b]) / n
    y_pred_b = torch.sum(y_pred_true[indices_b]) / n
    out = torch.abs(y_pred_b - y_conf_b) + n
    return out * n


def get_confusion_matrix(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return


def plot_mce(df: torch.Tensor, save_path: str):
    # save fig in folder
    return


def plot_ece(df: torch.Tensor, save_path: str):
    # save fig in folder
    return


def plot_confusion_matrix(cm: torch.Tensor, save_path: str, cmap=plt.cm.gray_r,
                          benchmark_session_id: str = ""):
    return


def get_y_conf_true(y_conf_2d, y_true_1d):
    """

    :param y_conf_2d: shape (batch_size, n_class)
    :param y_true_1d: 1d true labels
    :return: 1d vec of confidence score for true labels only
    """
    batch_size = len(y_true_1d)
    indices = torch.stack((torch.arange(batch_size), y_true_1d))
    out = y_conf_2d[indices]
    return out


def get_y_pred_true(y_pred_2d, y_true_1d):
    """

    :param y_pred_2d: one-hot vectors of length batch_size.
    :param y_true_1d: 1d true labels
    :return: 1d vec of predicted accuracy for true labels only
    """
    y_pred_1d = torch.argmax(y_pred_2d, dim=1)
    out = y_pred_1d.eq(y_true_1d).nonzero().flatten()
    return out


