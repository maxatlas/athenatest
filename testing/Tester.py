import config as c
from utils import get_MNIST_test_set, get_dataloader
from testing.models import Model

MNIST_path = "../%s" % c.data_folder


class Tester:
    def __init__(self, model=Model(), data_path=MNIST_path, batch_size: int = c.batch_size_test,
                 shuffle: bool = c.shuffle_dataloader):
        self.model = model
        data = get_dataloader(get_MNIST_test_set(data_path), batch_size=batch_size, shuffle=shuffle)
        self.data = enumerate(data)

    def next(self):
        _, (X, y) = next(self.data)
        return X, y, *self.model.batch_eval(X)


