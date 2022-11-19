import config as c
from utils import get_MNIST_test_set, get_dataloader, batch_eval
from testing.models import CNN

MNIST_path = "../%s" % c.data_folder_path


class Tester:
    def __init__(self, model=CNN(), data_path=MNIST_path, batch_size: int = c.batch_size_test,
                 shuffle: bool = c.shuffle_dataloader):
        self.model = model
        data = get_dataloader(get_MNIST_test_set(data_path), batch_size=batch_size, shuffle=shuffle)
        self.data = enumerate(data)

    def next(self):
        _, (X, y) = next(self.data)
        return X, y, *batch_eval(self.model, X)


