import torch
import matplotlib.pyplot as plt
import config as c
from pathlib import Path


def get_mce(y_conf: torch.Tensor, y_true: torch.Tensor, benchmark_session_id: str = ""):
    mce = 0
    return mce


def get_ece(y_conf: torch.Tensor, y_true: torch.Tensor, benchmark_session_id: str = ""):
    ece= 0

    return ece


def get_confusion_matrix(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    n_class = len(y_true[0])
    cm = torch.zeros(n_class, n_class).int()
    y_pred, y_true = torch.argmax(y_pred, dim=1), torch.argmax(y_true, dim=1)
    for true_i, pred_i in zip(y_true, y_pred):
        cm[true_i][pred_i] += 1
    return cm


def plot_mce(df: torch.Tensor, save_path: str):
    # save fig in folder
    return


def plot_ece(df: torch.Tensor, save_path: str):
    # save fig in folder
    return


def plot_confusion_matrix(cm: torch.Tensor, save_path: str, cmap=plt.cm.gray_r,
                          benchmark_session_id: str = ""):
    dim_cm = cm.shape[0]
    plt.matshow(cm, cmap=cmap)
    plt.title("Confusion Matrix\nfor %s" % benchmark_session_id)
    plt.colorbar()
    tick_marks = range(dim_cm)
    plt.xticks(tick_marks, range(dim_cm))
    plt.yticks(tick_marks, range(dim_cm))
    plt.xlabel("True class")
    plt.ylabel("Predicted class")

    plt.savefig(save_path)

