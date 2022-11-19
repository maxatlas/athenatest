import argparse
import config as c
import torch
from pathlib import Path
from testing.models import CNN
from utils import (init_folders, batch_eval,
                   load_model, get_MNIST_test_set, get_dataloader,
                   get_session_id)
from metrics import (get_ce, get_ce_b, get_mce, get_ece, plot_ce,
                     get_confusion_matrix, plot_confusion_matrix,
                     init_confusion_matrix)
from improve_fp import save_FP_samples


# """Ask for inputs"""
# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset_id',
#                     '-d',
#                     type=str,
#                     required=True)
# parser.add_argument('--model_id',
#                     '-m',
#                     type=str,
#                     required=True)
# parser.add_argument('--benchmark_session_id',
#                     '-b',
#                     type=str,
#                     required=False,
#                     default="",
#                     help="The session to compare the results with.")
# parser.add_argument('--min_size',
#                     '-s',
#                     type=int,
#                     required=False,
#                     default=c.min_n,
#                     help="Minimum sample size to testing deviation from benchmarking result.")
# parser.add_argument('--dev_thresh',
#                     '-t',
#                     type=float,
#                     required=False,
#                     default=c.deviation_threshold,
#                     help="Deviation threshold from benchmark results before termination.")
# args = parser.parse_args()
#
# model_id, dataset_id = args.model_ids, args.dataset_id
# session_id = get_session_id(model_id, dataset_id)
#
# benchmark_session_id, min_size, dev_thresh = args.benchmark_session_id, args.min_size, args.dev_thresh

# print("Loading models %s and dataset %s..." % (model_id, dataset_id))
#
init_folders()
# print("Folders created.")
#
"""Load models and dataset per identifiers"""
model = CNN()
print("\nModel %s loaded." % "")
test_loader = get_dataloader(get_MNIST_test_set(c.data_folder_path))
dataset_size = len(test_loader.dataset)
print("\nDataset %s loaded." % "MNIST")
n_class = len(test_loader.dataset.classes)

"""Evaluate model"""
eval_res = {}
ce, mce, cm = torch.zeros(c.k), -1, init_confusion_matrix(n_class)
for batch_id, (X, y) in enumerate(test_loader):
    y_pred, y_conf = batch_eval(model, X)
    ce_b = get_ce_b(y_pred, y_conf, y, c.k)
    mce_cur = get_mce(ce_b)
    mce = mce_cur if mce_cur > mce else mce
    ce += get_ce(ce_b)
    cm = get_confusion_matrix(y, y_pred, n_class=n_class, cm_last=cm)

    fp_samples = save_FP_samples(batch_id, test_loader.dataset.classes, X,
                                 y_pred_1d=y_pred, y_true_1d=y,
                                 save_path=c.fp_folder_path)

ce = torch.abs(ce)
ece = get_ece(ce, batch_size=dataset_size)

pass
"""Store results to no/sql database/ whatever logging system in place."""
plot_ce(ce, Path(c.res_folder_path)/"calibration_graph.png")
plot_confusion_matrix(cm, Path(c.res_folder_path)/"confusion_matrix.png")

"""Model Improvement"""
# Model improvement only works on single model evaluation session.
# primitive idea: dimension reduction + any clustering method

