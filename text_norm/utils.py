import argparse
import errno
import os
import re
import typing
from datetime import datetime

import numpy as np
from text_norm.data_set import DataSet


DataSets = typing.NamedTuple('DataSets', [('train', DataSet), ('validation', DataSet), ('test', DataSet)])


def root_dir(path: str = ''):
    res_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if path:
        res_path = os.path.join(res_path, path)

    return res_path


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


def eval_expr(expr: str, values: dict) -> int:
    for key in values:
        expr = expr.replace(key, str(values[key]))

    if not re.match('^[-+*/^.()0-9]*$', expr):
        raise ValueError('Invalid expression')

    return int(eval(expr))


def get_session_id():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=str, default=None, help='Session ID')

    args = parser.parse_args()
    session_id = args.session  # use a particular session ID, it restores existing session or creates new one

    if not session_id:
        session_id = datetime.today().strftime('%y%m%d_%H%M')

    return session_id
