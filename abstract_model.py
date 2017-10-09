import tensorflow as tf
import logging
from abc import ABC


class AbstractModel(ABC):
    def __init__(self):
        logger = logging.getLogger(__name__)
        logger.debug('Building the model')

        # create a graph
        self._graph = tf.Graph()

    @property
    def graph(self):
        return self._graph
