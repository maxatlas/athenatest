import torch
import torchvision
import config as c
from pathlib import Path


torch.backends.cudnn.enabled = False
torch.manual_seed(c.random_seed)


def get_MNIST_test_set():
    test_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST(str(Path(__file__).parent.parent/'data'), train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,)),
                                 ]),
                                 target_transform=torchvision.transforms.Compose([
                                     lambda x: torch.tensor(x), ]),
                                 ),
      batch_size=c.batch_size_test, shuffle=True)
    return test_loader

