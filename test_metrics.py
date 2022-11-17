import torch
import torchvision
from metrics import *
import torch.nn.functional as F


batch_size_test = 100

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,)),
                             ]),
                             target_transform=torchvision.transforms.Compose([
                                 lambda x: torch.tensor(x),
                                 lambda x: F.one_hot(x, 10)]),
                             ),
  batch_size=batch_size_test, shuffle=True)

tester = enumerate(test_loader)
_, (_, y_1) = next(tester)
_, (_, y_2) = next(tester)


def test_cm():
    cm = get_confusion_matrix(y_1, y_2)
    print(cm)
    cm = get_confusion_matrix(y_1, y_1)
    print(cm)
    return


def test_plot_cm():
    cm = get_confusion_matrix(y_1, y_2)
    plot_confusion_matrix(cm, "vis/cm.png", benchmark_session_id="model_v1+MNIST")


if __name__ == "__main__":
    test_plot_cm()
    # print(test_cm())
