from improve_fp import *
from Tester import Tester
import config as c
import torch

tester = Tester()
X, y_1, y_pred, y_conf = tester.next()


def test_get_FP_samples():
    X = torch.tensor([[1], [2], [3]])
    y_pred = torch.tensor([4, 7, 9])
    y_true = torch.tensor([2, 7, 10])
    assert get_FP_samples(9, X, y_pred, y_true)[0].tolist() == [[[3]]], \
        "wrong output for get_FP_samples"


def test_save_FP_samples():
    for i in range(c.n_class):
        fp_X, fp_y = get_FP_samples(i, X, y_pred_1d=y_pred, y_true_1d=y_1)
        save_FP_samples(fp_X, fp_y, "../results/false_positives"), \
            "wrong operation for save_FP_samples"


if __name__ == "__main__":
    test_get_FP_samples()
    test_save_FP_samples()

