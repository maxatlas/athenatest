import config as c
from utils import load_dataset
from testing.models import Model

MNIST_path = "../%s" % c.data_folder
mnist = load_dataset(MNIST_path)   # less is correctly predicted. Lower error.


class Tester:
    def __init__(self, model=Model(), data=mnist):
        self.model = model
        self.data = enumerate(data)

    def next(self):
        _, (X, y) = next(self.data)
        return X, y, *self.model.batch_eval(X)


