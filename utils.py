import os
import errno
import typing
import numpy as np
from data_set import DataSet

DataSets = typing.NamedTuple('DataSets', [('train', DataSet), ('validation', DataSet), ('test', DataSet)])


def check_path(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise


def get_one_hot_matrix(num_vectors: int, vector_dim: int, ids: list) -> np.ndarray:
    one_hot_matrix = np.zeros((num_vectors, vector_dim))
    one_hot_matrix[np.arange(len(ids)), ids] = 1

    return one_hot_matrix


def log_loss(labels, predictions):
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
    return (labels - 1) * np.log(1 - predictions) - labels * np.log(predictions)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
