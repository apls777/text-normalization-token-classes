import tensorflow as tf
from tensorflow.contrib import rnn
from abstract_model import AbstractModel


class PredictionModel(AbstractModel):
    def __init__(self,
                 char_dim: int,
                 token_dim: int,
                 num_chars: int,  # number of characters in a token
                 num_classes: int,  # number of possible token classes
                 batch_size: int,  # number of tokens in a batch
                 num_token_dependencies: int,  # token class depends on X tokens to the left
                 token_num_layers: int = 1):

        super().__init__()

        # build the model
        with self._graph.as_default():
            self._tokens = tf.placeholder(tf.uint8, shape=(batch_size, num_chars), name='tokens')

            # convert indices to one-hot vectors
            one_hot_tokens = tf.one_hot(self._tokens, char_dim, axis=-1,
                                        name='tokens_one_hot')  # dim: [batch_size, num_chars, char_dim]

            # init LSTM cells
            def single_cell():
                return rnn.BasicLSTMCell(token_dim)

            if token_num_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(token_num_layers)])
            else:
                cell = single_cell()

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
            self._raw_predictions = tf.matmul(x, W) + b  # dim: [batch_size, num_classes]

            # predictions
            self._predictions = tf.argmax(self._raw_predictions, 1)  # dim: [batch_size]

    def get_predictions(self, sess, data):
        feed_dict = {self._tokens: data}
        return sess.run(self._predictions, feed_dict=feed_dict)
