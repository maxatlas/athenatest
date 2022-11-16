import os
import config as c
import shutil

from pathlib import Path
from datetime import datetime


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
    :param timestamp:
    :return:
    """
    session_id = "%s%s%s" % (raw_model_ids, c.delimiter, dataset_id)
    return session_id



