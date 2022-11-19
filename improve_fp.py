import torch
import os

from pathlib import Path
from metrics import get_y_pred_true
from torchvision.utils import save_image


def get_FP_samples(class_i: int, X: torch.Tensor, y_pred_1d: torch.Tensor, y_true_1d: torch.Tensor):
    """
    :param class_i:
    :param X:
    :param y_pred_1d:
    :param y_true_1d:
    :return:
    """
    # get mask for falsely classified
    incorrect_ids = ~get_y_pred_true(y_pred_1d, y_true_1d).flatten().bool()
    # get mask for classified as class_i
    y_pred_i = (y_pred_1d == class_i).bool()
    # get the overlap indices
    fp_ids = torch.mul(incorrect_ids, y_pred_i).nonzero()

    samples = X[fp_ids, :]
    y_true = y_true_1d[fp_ids]

    return samples, y_true


def save_FP_samples(samples_X: torch.Tensor, samples_y: torch.Tensor, save_path: str):
    """
    :param samples_X:
    :param samples_y:
    :param save_path:
    :return:
    """
    save_path = Path(save_path)
    if samples_X.shape[0] > 0:
        for i in range(len(samples_X)):
            label = str(samples_y[i].tolist()[0])
            os.makedirs(save_path/label, exist_ok=True)
            save_image(samples_X[i], save_path/label/("%i.png" % i))