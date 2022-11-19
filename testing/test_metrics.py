from Tester import Tester
from metrics import *


tester = Tester()
_, y_1, y_pred, y_conf = tester.next()
_, y_2, _, _ = tester.next()


def test_get_y_conf_1d():
    y_conf_2d = torch.tensor([[0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])
    y_pred_1d = torch.argmax(y_conf_2d, dim=1)
    out = get_y_conf_1d(y_conf_2d, y_pred_1d).tolist()
    out = [round(i, 1) for i in out]
    assert out == [0.8, 0.6, 0.9]


def test_get_y_pred_true():
    y_conf_2d = torch.tensor([[0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])
    y_true_1d = torch.tensor([0, 1, 1])
    y_pred_1d = torch.argmax(y_conf_2d, dim=1)
    out = get_y_pred_true(y_pred_1d, y_true_1d).tolist()
    assert out == [0, 0, 1]


def test_get_ece():
    bin_size = 3

    y_pred_1d = torch.tensor([0, 2, 2, 4, 0, 1, 1, 3, 3, 4])
    y_true_1d = torch.tensor([0, 2, 3, 4, 2, 0, 1, 3, 3, 2])
    y_conf_2d = torch.tensor([
        [0.25, 0.20, 0.22, 0.18, 0.15],
        [0.16, 0.06, 0.50, 0.07, 0.21],
        [0.06, 0.03, 0.80, 0.07, 0.04],
        [0.02, 0.03, 0.01, 0.04, 0.90],
        [0.40, 0.15, 0.16, 0.14, 0.15],
        [0.15, 0.28, 0.18, 0.17, 0.22],
        [0.07, 0.80, 0.03, 0.06, 0.04],
        [0.10, 0.05, 0.03, 0.75, 0.07],
        [0.25, 0.22, 0.05, 0.30, 0.18],
        [0.12, 0.09, 0.02, 0.17, 0.60],
    ])

    ece, mce, ce = get_ece_mce_ce(y_pred_1d, y_conf_2d, y_true_1d, bin_size=bin_size)

    plot_ce(ce, save_path="../vis/ce.png", bin_size=bin_size)

    assert round(float(ece), 3) == 0.192
    assert round(float(mce), 3) == 0.390

    return mce, ece


def test_cm():
    class_size = 3
    y_pred = torch.Tensor([0, 2, 1, 0, 0, 2, 2, 1, 1, 1])
    y_true = torch.Tensor([2, 2, 0, 0, 0, 1, 0, 0, 1, 1])

    cm = get_confusion_matrix(y_pred, y_true, n_class=class_size)
    assert cm.tolist() == [[2, 2, 1],
                           [0, 2, 1],
                           [1, 0, 1]]
    cm = get_confusion_matrix(y_pred, y_pred, n_class=class_size)
    assert cm.tolist() == [[3, 0, 0],
                           [0, 4, 0],
                           [0, 0, 3]]

    cm = get_confusion_matrix(torch.tensor([]), torch.tensor([]), n_class=class_size)
    assert cm.tolist() == torch.zeros(class_size, class_size).int().tolist()

    test_plot_cm(cm)
    return


def test_plot_cm(cm):
    plot_confusion_matrix(cm, "../vis/cm.png", benchmark_session_id="model_v1+MNIST")


if __name__ == "__main__":
    test_get_y_conf_1d()
    test_get_y_pred_true()
    test_cm()
    test_get_ece()

