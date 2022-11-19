from utils import *
import config as c


def test_load_dataset():
    ds = get_test_set(".."/Path(c.data_folder_path), pad=c.pad)
    dl = enumerate(get_dataloader(ds))
    i, (X, y) = next(dl)
    assert i == 0
    assert y.shape[0] == 100


if __name__ == "__main__":
    test_load_dataset()

