from Tester import Tester
from metrics import *

tester = Tester()
_, y_1, y_pred, y_conf = tester.next()
_, y_2, y_pred_2, y_conf_2 = tester.next()


def test_get_y_conf_1d():
    y_conf_2d = torch.tensor([[0.2, 0.8],
                              [0.6, 0.4],
                              [0.1, 0.9]])
    y_pred_1d = torch.argmax(y_conf_2d, dim=1)
    out = get_y_conf_1d(y_conf_2d, y_pred_1d).tolist()
    out = [round(i, 1) for i in out]
    assert out == [0.8, 0.6, 0.9], "wrong output for metrics.get_y_conf_1d()"


def test_get_y_pred_true():
    y_conf_2d = torch.tensor([[0.2, 0.8],
                              [0.6, 0.4],
                              [0.1, 0.9]])
    y_true_1d = torch.tensor([0, 1, 1])
    y_pred_1d = torch.argmax(y_conf_2d, dim=1)
    out = get_y_pred_true(y_pred_1d, y_true_1d).tolist()
    assert out == [0, 0, 1], "wrong output for metrics.get_y_pred_true()"


def test_get_ece():
    bin_size = torch.tensor(3)

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

    batch_size = len(y_pred_1d)
    ce_b = get_ce_b(y_pred_1d, y_conf_2d, y_true_1d, bin_size=bin_size)
    ce = get_ce(ce_b, absolute=True)

    ece = get_ece(ce, batch_size=batch_size)
    mce = get_mce(ce_b)

    plot_ce(ce, batch_size=batch_size, save_path=Path(c.test_output_folder)/"ce.png", bin_size=bin_size)

    assert round(float(ece), 3) == 0.192, "wrong ece"
    assert round(float(mce), 3) == 0.390, "wrong mce"

    return mce, ece


def test_cm():
    class_size = 3
    y_pred = torch.Tensor([0, 2, 1, 0, 0, 2, 2, 1, 1, 1])
    y_true = torch.Tensor([2, 2, 0, 0, 0, 1, 0, 0, 1, 1])

    cm = get_confusion_matrix(y_pred, y_true, n_class=class_size)
    assert cm.tolist() == [[2, 2, 1],
                           [0, 2, 1],
                           [1, 0, 1]], "wrong confusion matrix computation"
    test_plot_cm(cm, "cm_normal")

    cm = get_confusion_matrix(y_pred, y_pred, n_class=class_size)
    assert cm.tolist() == [[3, 0, 0],
                           [0, 4, 0],
                           [0, 0, 3]], "wrong confusion matrix computation"
    test_plot_cm(cm, "cm_all_correct")

    cm = get_confusion_matrix(torch.tensor([]), torch.tensor([]), n_class=class_size)
    assert cm.tolist() == torch.zeros(class_size, class_size).int().tolist(), \
        "wrong confusion matrix computation"
    test_plot_cm(cm, "cm_empty")

    return


def test_plot_cm(cm, file_name: str = "cm"):
    plot_confusion_matrix(cm, Path(c.test_output_folder)/("%s.png" % file_name))


def test_agg_ce():
    b1, b2 = 100, 19
    b = b1 + b2
    tester1 = Tester(batch_size=b1)
    tester2 = Tester(batch_size=b2)
    _, y_true_1, y_pred_1, y_conf_1 = tester1.next()
    _, y_true_2, y_pred_2, y_conf_2 = tester2.next()
    y_true = torch.concat([y_true_1, y_true_2], dim=0)
    y_pred = torch.concat([y_pred_1, y_pred_2], dim=0)
    y_conf = torch.concat([y_conf_1, y_conf_2], dim=0)

    ce = get_ce_b(y_pred, y_conf, y_true)
    ce_1 = get_ce_b(y_pred_1, y_conf_1, y_true_1)
    ce_2 = get_ce_b(y_pred_2, y_conf_2, y_true_2)

    ce = get_ce(ce)
    ce_1 = get_ce(ce_1)
    ce_2 = get_ce(ce_2)
    ce_agg = (ce_1 + ce_2)

    plot_ce(ce, batch_size=b, save_path="%s/ce_MNIST_agg.png" % c.test_output_folder)
    plot_ce(ce_1, batch_size=b1, save_path="%s/ce_MNIST_1.png" % c.test_output_folder)
    plot_ce(ce_1, batch_size=b2, save_path="%s/ce_MNIST_2.png" % c.test_output_folder)

    ce = torch.abs(ce)
    ce_agg = torch.abs(ce_agg)
    ece = get_ece(ce, b1+b2)
    ece_agg = get_ece(ce_agg, b1+b2)

    assert all(torch.isclose(ce, ce_agg))
    assert ece.isclose(ece_agg)


if __name__ == "__main__":
    test_agg_ce()
    test_get_y_conf_1d()
    test_get_y_pred_true()
    test_cm()
    test_get_ece()

