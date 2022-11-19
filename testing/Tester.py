import torch
import config as c
from utils import load_dataset
from testing.models import Model
from testing.dataset import get_MNIST_test_set
from pathlib import Path

MNIST_path = "../"/Path(c.data_folder)
data = get_MNIST_test_set()  # more is. Higher error because confidence score is the same.
data = load_dataset(MNIST_path)   # less is correctly predicted. Lower error.


class Tester:
    def __init__(self, model=Model(), data=data):
        self.model = model
        self.data = enumerate(data)

    def next(self):
        _, (X, y) = next(self.data)
        return X, y, *self.model.batch_eval(X)


