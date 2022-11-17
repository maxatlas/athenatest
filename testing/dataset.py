import torch
import torchvision
import torch.nn.functional as F
import config as c


"""Dataset"""
torch.backends.cudnn.enabled = False
torch.manual_seed(c.random_seed)


def get_MNIST_test_set():
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
      batch_size=c.batch_size_test, shuffle=True)

    tester = enumerate(test_loader)
    batch_idx, (X, y) = next(tester)
    return X, y

