import os
import config as c
import torch

from pathlib import Path
from metrics import get_y_pred_true


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


def load_model(model_id: str):
    """
    This is a pseudo function that should return model based on some database lookup or API communication.
    :param model_id: string
    :return:
    """
    return


def load_dataset(dataset_id: str):
    """
    This is a pseudo function that should return dataset based on some database lookup or API communication.
    :param dataset_id: string
    :return:
    """
    return


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


def get_FP_samples(X, y_pred_1d, y_true_1d, save_path):
    correct_ids = get_y_pred_true(y_pred_1d, y_true_1d)
    samples = X[correct_ids, :]


