from Tester import Tester
from utils import get_FP_samples, save_FP_samples


tester = Tester()
X, y_1, y_pred, y_conf = tester.next()


def test_save_FP_samples():
    fp = get_FP_samples(X, y_pred_1d=y_pred, y_true_1d=y_1)
    save_FP_samples(fp, "../FP")


if __name__ == "__main__":
    test_save_FP_samples()
