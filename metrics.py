import torch
import matplotlib.pyplot as plt
import config as c
from pathlib import Path


def get_mce(y_conf: torch.Tensor, y_true: torch.Tensor, benchmark_session_id: str = ""):
    mce = 0
    return mce


def get_ece(y_conf: torch.Tensor, y_pred, y_true: torch.Tensor, benchmark_session_id: str = ""):
    ece= 0

    return ece




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


def get_