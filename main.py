import argparse
import config as c
import torch
from pathlib import Path
from utils import (init_folders, batch_eval,
                   load_model, get_test_set, get_dataloader)
from metrics import (get_ce, get_ce_b, get_mce, get_ece, plot_ce,
                     get_confusion_matrix, plot_confusion_matrix,
                     init_confusion_matrix)
from improve_fp import save_FP_samples


"""Ask for inputs"""
parser = argparse.ArgumentParser()
parser.add_argument('--data',
                    '-d',
                    type=str,
                    default=c.data_folder_path)
parser.add_argument('--model',
                    '-m',
                    type=str,
                    default="tiny")
parser.add_argument('--device',
                    type=str,
                    default="cpu")
parser.add_argument("--save_fp",
                    "-s",
                    type=bool,
                    default=False)
parser.add_argument('--dev_thresh',
                    '-t',
                    type=float,
                    default=c.deviation_threshold,
                    help="Deviation threshold from benchmark results before termination.")
args = parser.parse_args()

model_i, data_path = args.model, args.data

device, save_fp, dev_thresh = args.device, args.save_fp, args.dev_thresh

init_folders()
print("Folders created.")


"""Load models and dataset per identifiers"""
test_loader = get_dataloader(get_test_set(c.data_folder_path, pad=2))
n_class = len(test_loader.dataset.classes)
dataset_size = len(test_loader.dataset)
print("\nDataset loaded from %s." % data_path)
model = load_model(model_i, n_class).to(device)
print("\nConvNext %s loaded." % model_i)

bin_size = torch.tensor(c.k).to(device)

"""Evaluate model"""
eval_res = {}
ce, mce, cm = torch.zeros(c.k), -1, init_confusion_matrix(n_class)
for batch_id, (X, y) in enumerate(test_loader):
    X, y = X.to(device), y.to(device)
    y_pred, y_conf = batch_eval(model, X)
    # get list of CE value and bucket_size pair
    ce_b = get_ce_b(y_pred, y_conf, y, bin_size)
    # get current MCE value
    mce_cur = get_mce(ce_b)
    mce = mce_cur if mce_cur > mce else mce
    # get list of product of CE and bucket_size
    ce += get_ce(ce_b)

    cm = get_confusion_matrix(y, y_pred, n_class=n_class, cm_last=cm)
    if save_fp:
        fp_samples = save_FP_samples(batch_id, test_loader.dataset.classes, X,
                                     y_pred_1d=y_pred, y_true_1d=y,
                                     save_path=c.fp_folder_path)

ece = get_ece(ce, batch_size=dataset_size)


"""Store results to no/sql database/ whatever logging system in place."""
plot_ce(ce, batch_size=dataset_size, bin_size=bin_size, save_path=Path(c.res_folder_path)/"calibration_graph.png")
plot_confusion_matrix(cm, Path(c.res_folder_path)/"confusion_matrix.png")


"""Model Improvement"""
# Model improvement only works on single model evaluation session.
# primitive idea: dimension reduction + any clustering method

