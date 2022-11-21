from improve_fp import *
from Tester import Tester
import config as c
import torch
import os
os.makedirs(c.test_fp_folder, exist_ok=True)

tester = Tester()
X, y_1, y_pred, y_conf = tester.next()


def test_get_FP_samples():
    X = torch.tensor([[1], [2], [3]])
    y_pred = torch.tensor([4, 7, 9])
    y_true = torch.tensor([2, 7, 10])
    assert get_FP_samples_per_class(9, X, y_pred, y_true)[0].tolist() == [[[3]]], \
        "wrong output for get_FP_samples"


def test_save_FP_samples():
    save_FP_samples(0, range(10), X[:3], y_pred[:c.testing_limit], y_1[:c.testing_limit],
                    c.fp_folder_path), "wrong operation for save_FP_samples"


def test_associate_cluster_to_label():
    y_true = torch.tensor([1, 1, 0, 0, 3, 3, 2, 1])
    y_pred = torch.tensor([2, 2, 3, 3, 1, 1, 0, 2])
    out = associate_cluster_to_label(y_pred, y_true, n_cluster=4)
    assert out == {0: 2, 1: 3, 2: 1, 3: 0}


def test_model_eval():
    out = eval_model(".."/Path(c.fp_folder_path))
    print(out)


if __name__ == "__main__":
    # test_get_FP_samples()
    # test_save_FP_samples()
    # test_associate_cluster_to_label()
    test_model_eval()