import torch
import os

from pathlib import Path
from metrics import get_y_pred_true
from torchvision.utils import save_image


def save_FP_samples(batch_i: int, classes, X, y_pred_1d: torch.Tensor, y_true_1d: torch.Tensor, save_path: str = ""):
    for label in classes:
        label = int(label)
        X_c, y_c = get_FP_samples_per_class(label, X, y_pred_1d, y_true_1d)
        save_path = Path(save_path)
        if X_c.shape[0] > 0:
            for i in range(len(X_c)):
                label = str(y_c[i].tolist()[0])
                os.makedirs(save_path / label, exist_ok=True)
                save_image(X_c[i], save_path / label / ("%i%i.png" % (batch_i, i)))


def get_FP_samples_per_class(class_i: int, X: torch.Tensor, y_pred_1d: torch.Tensor, y_true_1d: torch.Tensor):
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


