import os
import config as c
import torch
from pathlib import Path
from PIL import Image

# for dataset loading and saving
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


torch.manual_seed(c.random_seed)


def init_folders():
    """
    This script simulates database environment with file system.
    Create needed folders.
    :return:
    """
    folders = [c.fp_folder_path, c.data_folder_path]
    for folder in folders:
        os.makedirs(Path(folder), exist_ok=True)


def load_model(model: str):
    """

    :param model: string
    :return:
    """
    return


def get_MNIST_test_set(folder_path: Path):
    """

    :param folder_path: string
    :return:
    """
    dataset = datasets.ImageFolder(str(folder_path), transform=transforms.Compose([
        transforms.ToTensor(),
    ]),
                                   target_transform=transforms.Compose([
                                       lambda x: torch.tensor(x), ]),
                                   )
    return dataset


def get_dataloader(dataset, batch_size: int = c.batch_size_test, shuffle: bool = c.shuffle_dataloader):
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dl


def load_img(img_path: str):
    img_path = Path(img_path)
    img = Image.open(img_path)
    converter = transforms.ToTensor()
    return converter(img)


def get_boundaries(bin_size: int = c.k):
    return torch.arange(0, 1.01, 1 / bin_size)


def batch_eval(model, x):
    model.eval()
    with torch.no_grad():
        logits = model.forward(x)
        probs = torch.sigmoid(logits)
        preds = torch.argmax(probs, dim=1)
    return preds, probs


