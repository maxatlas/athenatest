import argparse
import config as c

from utils import (init_folders,
                   load_model, load_dataset,
                   get_session_id)
from metrics import (get_mce, get_ece, get_confusion_matrix,
                     plot_mce, plot_ece, plot_confusion_matrix,)


"""Ask for inputs"""
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_id',
                    '-d',
                    type=str,
                    required=True)
parser.add_argument('--model_id',
                    '-m',
                    type=str,
                    required=True)
parser.add_argument('--benchmark_session_id',
                    '-b',
                    type=str,
                    required=False,
                    default="",
                    help="The session to compare the results with.")
parser.add_argument('--min_size',
                    '-s',
                    type=int,
                    required=False,
                    default=c.min_n,
                    help="Minimum sample size to test deviation from benchmarking result.")
parser.add_argument('--dev_thresh',
                    '-t',
                    type=float,
                    required=False,
                    default=c.deviation_threshold,
                    help="Deviation threshold from benchmark results before termination.")
args = parser.parse_args()

model_id, dataset_id = args.model_ids, args.dataset_id
session_id = get_session_id(model_id, dataset_id)

benchmark_session_id, min_size, dev_thresh = args.benchmark_session_id, args.min_size, args.dev_thresh

print("Loading models %s and dataset %s..." % (model_id, dataset_id))

init_folders(session_id)
print("Folders created.")

"""Load models and dataset per identifiers"""
model = load_model(model_id)
print("\nModel %s loaded." % model_id)
dataset = load_dataset(dataset_id)
print("\nDataset %s loaded." % dataset_id)


"""Evaluate model"""
eval_res = {}

y_predict = model.forward(dataset.X)
eval_res[c.mce_table_name] = get_mce(y_predict, dataset.y, benchmark_session_id)
eval_res[c.ece_table_name] = get_ece(y_predict, dataset.y, benchmark_session_id)
eval_res[c.cm_table_name] = get_confusion_matrix(y_predict, dataset.y, benchmark_session_id)


"""Store results to no/sql database/ whatever logging system in place."""
plot_mce(eval_res.get(c.mce_table_name))
plot_ece(eval_res.get(c.ece_table_name))
plot_confusion_matrix(eval_res.get(c.cm_table_name))


"""Model Improvement"""
# Model improvement only works on single model evaluation session.
# primitive idea: dimension reduction + any clustering method

