import torch
import os
import config as c
import utils
from pathlib import Path
from metrics import get_y_pred_true, get_accuracy, \
    get_confusion_matrix, plot_confusion_matrix
from torchvision.utils import save_image

from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans


def eval_model(fp_folder: Path = c.fp_folder_path):
    fp = utils.get_test_set(fp_folder, pad=c.pad)
    if len(fp) == 0:
        return "N/A", "N/A"

    fp = utils.get_dataloader(fp, batch_size=len(fp))

    n_class = len(fp.dataset.classes)
    model = Model(n_class=n_class, n_cluster=c.n_cluster)

    _, (X, y) = next(enumerate(fp))
    y_pred = model(X, y)

    acc = get_accuracy(y_pred, y)
    cm = get_confusion_matrix(y, y_pred, n_class=n_class)
    plot_confusion_matrix(cm, Path(fp_folder).parent/"confusion_matrix_improved.png")

    return acc, cm


class Model(torch.nn.Module):
    def __init__(self, n_class: int, n_cluster: int):
        super(Model, self).__init__()
        self.n_class = n_class
        self.n_cluster = n_cluster
        self.tsne = TSNE(n_components=3, init="pca", learning_rate="auto")
        self.kmeans = MiniBatchKMeans(n_clusters=n_cluster)
        self.cluster_label_lookup = {}

    def forward(self, X, y):
        assert X.dim() == 4

        y_pred = torch.zeros(len(X))
        X = X.reshape(len(X), -1)
        X = self.tsne.fit_transform(X)
        self.kmeans.fit(X)
        y_cluster = torch.tensor(self.kmeans.labels_)
        self.cluster_label_lookup = associate_cluster_to_label(y_cluster, y, self.n_cluster)

        for i in range(len(X)):
            y_pred[i] = self.cluster_label_lookup[int(y_cluster[i])]

        assert len(y_pred.unique()) <= self.n_class

        return y_pred


def save_FP_samples(batch_i: int, classes, X, y_pred_1d: torch.Tensor, y_true_1d: torch.Tensor, save_path: str = ""):
    for i, label in enumerate(classes):
        X_c, y_c = get_FP_samples_per_class(torch.tensor(i).to(X.device), X, y_pred_1d, y_true_1d)
        save_path = Path(save_path)
        if X_c.shape[0] > 0:
            for j in range(len(X_c)):
                os.makedirs(save_path / str(label), exist_ok=True)
                save_image(X_c[j], save_path / str(label) / ("%i%i.png" % (batch_i, j)))


def get_FP_samples_per_class(class_i: int, X: torch.Tensor, y_pred_1d: torch.Tensor, y_true_1d: torch.Tensor):
    """
    :param class_i:
    :param X:
    :param y_pred_1d:
    :param y_true_1d:
    :return:
    """
    # get mask for falsely classified
    incorrect_ids = ~get_y_pred_true(y_pred_1d, y_true_1d).flatten().bool()
    # get mask for classified as class_i
    y_pred_i = (y_pred_1d == class_i).bool()
    # get the overlap indices
    fp_ids = torch.mul(incorrect_ids, y_pred_i).nonzero()

    samples = X[fp_ids, :]
    y_true = y_true_1d[fp_ids]

    return samples, y_true


def associate_cluster_to_label(y_cluster_1d, y_true_1d, n_cluster: int):
    """
    For each cluster, find the argmax of true labels.
    :param y_cluster_1d:
    :param y_true_1d:
    :param n_cluster:
    :return:
    """
    assert y_cluster_1d.shape == y_true_1d.shape
    assert y_cluster_1d.dim() == 1
    cluster_label_lookup = {}

    for i in range(n_cluster):
        cluster_label_lookup[i] = int(torch.mode(y_true_1d[y_cluster_1d == i]).values)

    return cluster_label_lookup

