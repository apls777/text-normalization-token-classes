import tensorflow as tf
from tensorflow.contrib import rnn
import logging


class Model(object):
    def __init__(self,
                 learning_rate: float,
                 char_dim: int,
                 token_dim: int,
                 num_chars: int,  # number of characters in a token
                 num_classes: int,  # number of possible token classes
                 batch_size: int,  # number of sentences in a batch
                 num_token_dependencies: int,  # token class depends on X tokens to the left
                 token_num_layers: int = 1):

        logger = logging.getLogger(__name__)
        logger.debug('Building the model')

        self._graph = tf.Graph()

        with self._graph.as_default():
            self._tokens = tf.placeholder(tf.uint8, shape=(batch_size, num_chars), name='tokens')
            self._labels = tf.placeholder(tf.uint8, shape=(batch_size,), name='token_classes')
            self._global_step = tf.Variable(0, trainable=False)

            # init LSTM cells
            def single_cell():
                return rnn.BasicLSTMCell(token_dim)

            if token_num_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(token_num_layers)])
            else:
                cell = single_cell()

            # convert indices to one-hot vectors
            one_hot_tokens = tf.one_hot(self._tokens, char_dim, axis=-1,
                                        name='tokens_one_hot')  # dim: [batch_size, num_chars, char_dim]
            one_hot_labels = tf.one_hot(self._labels, num_classes, axis=-1,
                                        name='classes_one_hot')  # dim: [batch_size, num_classes]

            # prepare the data for LSTM
            init_state = cell.zero_state(batch_size, dtype=tf.float32)
            inputs = tf.unstack(tf.transpose(one_hot_tokens, [1, 0, 2]))

            # get vectorized tokens
            outputs, _state = rnn.static_rnn(cell, inputs, init_state, dtype=tf.float32)
            vectorized_tokens = outputs[num_chars - 1]  # dim: [batch_size, token_dim]

            # add two empty tokens to the beginning of the batch
            vectorized_tokens = tf.pad(vectorized_tokens,
                                       [[num_token_dependencies, 0], [0, 0]])  # dim: [batch_size + X, num_chars]

            # build full-connected layer
            x = []
            for i in range(0, batch_size):
                # dim: [num_token_dependencies + 1, token_dim]
                tokens_group = vectorized_tokens[i:i + num_token_dependencies + 1]
                x.append(tf.reshape(tokens_group, [token_dim * (num_token_dependencies + 1)]))

            x = tf.stack(x)  # dim: [batch_size, token_dim * (num_token_dependencies + 1)]

            W = tf.Variable(tf.zeros([token_dim * (num_token_dependencies + 1), num_classes]))
            b = tf.Variable(tf.zeros([num_classes]))
            y = tf.matmul(x, W) + b

            # predictions
            self._predictions = tf.argmax(y, 1)

            # cross entropy
            self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=y))

            # accuracy
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(one_hot_labels, 1))
            self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self._train = optimizer.minimize(self._cost, global_step=self._global_step)

            # summaries
            tf.summary.scalar('loss', self._cost)
            tf.summary.scalar('accuracy', self._accuracy)
            self._summary = tf.summary.merge_all()

            # initializer
            self.init = tf.global_variables_initializer()

    @property
    def graph(self):
        return self._graph

    def init_variables(self, sess: tf.Session):
        return sess.run(self.init)

    def train_step(self, sess: tf.Session, data, labels):
        feed_dict = {self._tokens: data, self._labels: labels}
        summary_val, _, global_step = sess.run([self._summary, self._train, self._global_step], feed_dict=feed_dict)
        return summary_val, global_step

    def evaluate_model(self, sess: tf.Session, data, labels):
        feed_dict = {self._tokens: data, self._labels: labels}
        return sess.run([self._cost, self._accuracy], feed_dict=feed_dict)

    def get_predictions(self, sess, data):
        feed_dict = {self._tokens: data}
        return sess.run(self._predictions, feed_dict=feed_dict)
