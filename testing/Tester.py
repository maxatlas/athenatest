from testing.dataset import get_MNIST_test_set
from testing.models import Model
import torch
import config as c

torch.manual_seed(c.random_seed)


class Tester:
    def __init__(self, model=Model(), data=get_MNIST_test_set()):
        self.model = model
        self.data = enumerate(data)

    def next(self):
        _, (X, y) = next(self.data)
        return X, y, *self.model.batch_eval(X)


