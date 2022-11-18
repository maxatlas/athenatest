from Tester import Tester
from metrics import *


tester = Tester()
_, y_1, y_pred, y_conf = tester.next()
_, y_2, _, _ = tester.next()


def test_get_ece(y_pred_1d, y_conf_2d, y_true_1d):
    y_pred_true = get_y_pred_true(y_pred_1d, y_true_1d)
    y_conf_true = get_y_conf_true(y_conf_2d, y_true_1d)

    boundaries = torch.arange(0, 1.01, 1/c.k)
    b_by_conf = torch.bucketize(y_conf_true, boundaries)
    ce = get_ce(b_by_conf, y_conf_true, y_pred_true)

    plot_ce(ce, boundaries, save_path="../vis/ce.png")

    mce = get_mce(ce)
    ece = get_ece(ce, y_true_1d)

    assert round(float(ece), 4) == 0.1793
    assert round(float(mce), 4) == 0.4407

    return mce, ece


def test_cm(verbatim=True):
    cm0 = init_confusion_matrix()
    cm = get_confusion_matrix(y_1, y_2, cm0)
    if verbatim: print(cm)
    cm = get_confusion_matrix(y_1, y_1, cm0)
    if verbatim: print(cm)
    test_plot_cm(cm)
    return


def test_plot_cm(cm):
    plot_confusion_matrix(cm, "../vis/cm.png", benchmark_session_id="model_v1+MNIST")


if __name__ == "__main__":
    test_get_ece(y_pred, y_conf, y_1)
    test_cm(verbatim=True)

