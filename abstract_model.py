import tensorflow as tf
import logging
from abc import ABC


class AbstractModel(ABC):
    def __init__(self, logger: logging.Logger = None):
        self._logger = logger if logger is not None else logging.getLogger(__name__)
        self._logger.debug('Building the model')

        # create a graph
        self._graph = tf.Graph()

    @property
    def graph(self):
        return self._graph
