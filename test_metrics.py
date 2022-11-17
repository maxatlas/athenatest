import torch
from testing.dataset import get_MNIST_test_set
from testing.models import Model
from metrics import *


X, y = get_MNIST_test_set()
model = Model()
model.eval()
y_pred, y_conf = model.batch_eval(X)


def test_get_ece(y_pred_1d, y_conf_2d, y_true_2d):
    y_true_1d = torch.argmax(y_true_2d, dim=1)
    y_pred_true = get_y_pred_true(y_pred_1d, y_true_1d)
    y_conf_true = get_y_conf_true(y_conf_2d, y_true_1d)

    boundaries = torch.arange(0, 1.01, 1/c.k)
    b_by_conf = torch.bucketize(y_conf_true, boundaries)
    ce = get_ce(b_by_conf, y_conf_true, y_pred_true)

    plot_ce(ce, boundaries, save_file="vis/ce.png")

    ece = get_ece(ce, y_true_1d)
    mce = get_mce(ce, y_true_2d)

    assert round(float(ece), 4) == 0.1801
    assert round(float(mce), 4) == 0.6946

    return mce, ece


if __name__ == "__main__":
    test_get_ece(y_pred, y_conf, y)

