
from utils import *
import config as c


def test_load_dataset():
    ds = load_dataset(".."/Path(c.data_folder))
    dl = enumerate(ds)
    i, (X, y) = next(dl)
    assert i == 0
    assert X.shape == (100, 3, 28, 28)
    assert y.shape[0] == 100


if __name__ == "__main__":
    test_load_dataset()

