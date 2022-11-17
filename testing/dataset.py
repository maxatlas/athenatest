import torch
import torchvision
import torch.nn.functional as F


"""Dataset"""
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]),
                             target_transform=torchvision.transforms.Compose([
                                 lambda x: torch.tensor(x),
                                 lambda x: F.one_hot(x, 10)]),),
  batch_size=batch_size_train, shuffle=True)

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

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)


