import os
import config as c
import torch
from pathlib import Path
from PIL import Image

# for dataset loading and saving
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


torch.manual_seed(c.random_seed)


def init_folders(session_id):
    """
    This script simulates database environment with file system.
    Create needed folders.
    :return:
    """
    folders = [c.res_folder_name, c.FP_folder_name]
    os.makedirs(session_id, exist_ok=True)
    for folder in folders:
        os.makedirs(session_id/Path(folder), exist_ok=True)


def load_model(model: str):
    """

    :param model: string
    :return:
    """
    return


def load_dataset(folder_path: str):
    """

    :param folder_path: string
    :return:
    """
    dataset = datasets.ImageFolder(folder_path, transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ]),
                                   target_transform=transforms.Compose([
                                       lambda x: torch.tensor(x), ]),
                                   )
    dataset = DataLoader(dataset, batch_size=c.batch_size_test, shuffle=c.shuffle_dataloader)

    return dataset


def load_img(img_path: str):
    img_path = Path(img_path)
    img = Image.open(img_path)
    converter = transforms.ToTensor()
    return converter(img)


def get_session_id(raw_model_ids: str, dataset_id: str):
    """
    Get session id by hashing the concatenated string of raw_model_ids
    Calculate timestamp if not given.
    :param raw_model_ids:
    :param dataset_id:
    :return:
    """
    session_id = "%s%s%s" % (raw_model_ids, c.delimiter, dataset_id)
    return session_id


def get_boundaries(bin_size: int = c.k):
    return torch.arange(0, 1.01, 1 / bin_size)



