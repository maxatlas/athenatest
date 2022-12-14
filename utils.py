import os
import config as c
import torch
from pathlib import Path
from PIL import Image

# for dataset loading and saving
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


torch.manual_seed(c.random_seed)


def init_folders(folders=(c.fp_folder_path, c.data_folder_path)):
    """
    Create needed folders.
    :return:
    """
    for folder in folders:
        os.makedirs(Path(folder), exist_ok=True)


def load_model(model: str, n_class: int):
    """
    :param model: string
    :param n_class: int
    :return:
    """
    if model == "tiny":
        from torchvision.models import convnext_tiny
        model = convnext_tiny
    elif model == "small":
        from torchvision.models import convnext_small
        model = convnext_small
    elif model == "base":
        from torchvision.models import convnext_base
        model = convnext_base
    elif model == "large":
        from torchvision.models import convnext_large
        model = convnext_large
    else:
        raise ValueError("Model indicator not supported.")
    model = model(num_classes=n_class)
    return model


def get_test_set(folder_path: Path, pad: int) -> datasets.ImageFolder:
    """
    :param folder_path: string
    :param pad: int
    :return:
    """
    dataset = datasets.ImageFolder(str(folder_path), transform=transforms.Compose([
        transforms.Pad(pad),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,), (0.3081,)
        ),
    ]),
                                   target_transform=transforms.Compose([
                                       lambda x: torch.tensor(x), ]),
                                   )
    return dataset


def get_dataloader(dataset: datasets.ImageFolder, batch_size: int = c.batch_size_test,
                   shuffle: bool = c.shuffle_dataloader):
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dl


def load_img(img_path: str):
    img_path = Path(img_path)
    img = Image.open(img_path)
    converter = transforms.ToTensor()
    return converter(img)


def batch_eval(model, x: torch.Tensor):
    model.eval()
    with torch.no_grad():
        logits = model.forward(x)
        probs = torch.sigmoid(logits)
        preds = torch.argmax(probs, dim=1)
    return preds, probs


