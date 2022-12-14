import argparse
import warnings
import config as c
import torch
import shutil
from pathlib import Path
from utils import (init_folders, batch_eval,
                   load_model, get_test_set, get_dataloader)
from metrics import (get_ce, get_ce_b, get_mce, get_ece, plot_ce,
                     get_confusion_matrix, plot_confusion_matrix, init_confusion_matrix,
                     get_accuracy)
from improve_fp import save_FP_samples, eval_model

shutil.rmtree(c.fp_folder_path, ignore_errors=True)


"""Ask for inputs"""
parser = argparse.ArgumentParser()
parser.add_argument('--data',
                    '-d',
                    type=str,
                    default=c.data_folder_path)
parser.add_argument('--model',
                    '-m',
                    type=str,
                    default=c.model)
parser.add_argument('--device',
                    type=str,
                    default=c.device)
parser.add_argument("--save_fp",
                    "-s",
                    type=bool,
                    default=c.save_fp)
parser.add_argument('--acc_thresh',
                    '-t',
                    type=float,
                    default=c.accuracy_threshold,
                    help="Accuracy threshold for early stopping. ")
args = parser.parse_args()

model_i, data_path = args.model, args.data
device, save_fp, acc_thresh = args.device, args.save_fp, args.acc_thresh

init_folders()

"""Load model and dataset per identifiers"""
test_loader = get_dataloader(get_test_set(c.data_folder_path, pad=c.pad))
n_class = len(test_loader.dataset.classes)
dataset_size = len(test_loader.dataset)
print("\nDataset loaded from %s." % data_path)
model = load_model(model_i, n_class).to(device)
print("\nModel ConvNext %s loaded.\n" % model_i)

bin_size, acc_thresh = torch.tensor(c.k).to(device), torch.tensor(acc_thresh).to(device)

"""Evaluate model"""
ce_b, cm, acc = torch.zeros(c.k, 2), init_confusion_matrix(n_class), 0

for batch_id, (X, y) in enumerate(test_loader):
    X, y = X.to(device), y.to(device)
    y_pred, y_conf = batch_eval(model, X)

    # early stopping
    acc = acc * c.batch_size_test + len(X) * get_accuracy(y_pred, y)
    acc = acc / (c.batch_size_test + len(X))

    if acc < acc_thresh:
        warnings.warn("Model accuracy below set threshold. Terminating evaluation now...")
        break

    # aggregate list of CE value and bucket_size pair
    ce_b += get_ce_b(y_pred, y_conf, y, bin_size)

    cm = get_confusion_matrix(y, y_pred, n_class=n_class, cm_last=cm)

    if save_fp:
        fp_samples = save_FP_samples(batch_id, test_loader.dataset.classes, X,
                                     y_pred_1d=y_pred, y_true_1d=y,
                                     save_path=c.fp_folder_path)
ce = get_ce(ce_b)
ece = get_ece(ce_b, batch_size=dataset_size)
mce = get_mce(ce)

print("ECE value:", float(ece))
print("MCE value:", float(mce))
print("Accuracy:", float(acc))

"""Plotting"""
plot_ce(ce, bin_size=bin_size, save_path=Path(c.res_folder_path)/"calibration_graph.png")
plot_confusion_matrix(cm, Path(c.res_folder_path)/"confusion_matrix.png")

print("\nPlots saved at", c.res_folder_path)

"""Model Improvement"""
print("\nImproving the false positives with model from improve_fp.py ...")
print("Processing with CPU. This could take a while ... \n")
# Model improvement only works on single model evaluation session.
# primitive idea: dimension reduction + any clustering method
acc, cm = eval_model()
print("Accuracy after improvement:", float(acc))
print("Confusion matrix after improvement is saved at %s." % c.res_folder_path)
