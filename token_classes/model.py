import tensorflow as tf
from tensorflow.contrib import rnn
import logging


class Model(object):
    def __init__(self,
                 learning_rate: float,
                 char_dim: int,
                 token_dim: int,
                 num_chars: int,  # number of characters in a token
                 num_words: int,  # number of words in a sentence
                 num_sentences: int,  # number of sentences in a batch
                 num_classes: int,  # number of possible token classes
                 token_num_layers: int = 1):

        logger = logging.getLogger(__name__)
        logger.debug('Building the model')

        self._graph = tf.Graph()

        with self._graph.as_default():
            # batch size
            batch_size = num_sentences * num_words

            self._tokens = tf.placeholder(tf.float32, shape=(batch_size, num_chars, char_dim), name='tokens')
            self._labels = tf.placeholder(tf.float32, shape=(batch_size, num_classes), name='token_classes')
            self._global_step = tf.Variable(0, trainable=False)

            # init LSTM cells
            def single_cell():
                return rnn.BasicLSTMCell(token_dim)

            if token_num_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(token_num_layers)])
            else:
                cell = single_cell()

            init_state = cell.zero_state(batch_size, dtype=tf.float32)
            inputs = tf.unstack(tf.transpose(self._tokens, [1, 0, 2]))

            # outputs dim: [batch_size, token_dim]
            outputs, state = rnn.static_rnn(cell, inputs, init_state, dtype=tf.float32)

            # dim: [num_sentences, num_words, token_dim]
            sentences = tf.split(outputs[num_chars - 1], num_or_size_splits=num_sentences, axis=0)

            # dim: [num_sentences, num_words + 2, token_dim]
            sentences = tf.pad(sentences, [[0, 0], [2, 0], [0, 0]])

            # dim: [batch_size, token_dim * 3]
            x = []
            for i in range(0, num_sentences):
                for j in range(0, num_words):
                    tokens = sentences[i][j:j + 3]  # dim: [3, token_dim]
                    x.append(tf.reshape(tokens, [token_dim * 3]))

            x = tf.stack(x)

            W = tf.Variable(tf.zeros([token_dim * 3, num_classes]))
            b = tf.Variable(tf.zeros([num_classes]))
            y = tf.matmul(x, W) + b

            # cross entropy
            self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._labels, logits=y))

            # accuracy
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self._labels, 1))
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
