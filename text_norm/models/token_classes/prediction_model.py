import logging
import tensorflow as tf
from tensorflow.contrib import rnn
from text_norm.abstract_model import AbstractModel
from text_norm import utils


class PredictionModel(AbstractModel):
    def __init__(self,
                 char_dim: int,
                 token_dim: int,
                 num_chars: int,  # number of characters in a token
                 num_classes: int,  # number of possible token classes
                 batch_size: int,  # number of tokens in a batch
                 num_tokens_left: int,  # token class depends on X tokens to the left
                 num_tokens_right: int,  # token class depends on X tokens to the right
                 token_num_layers: int = 1,
                 layers: tuple = ('i', 'o'),
                 logger: logging.Logger = None):

        super().__init__(logger)

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
            # dim: [batch_size + num_tokens_left + num_tokens_right, num_chars]
            vectorized_tokens = tf.pad(vectorized_tokens, [[num_tokens_left, num_tokens_right], [0, 0]])

            # build vectors for fully connected layers
            num_related_tokens = num_tokens_left + num_tokens_right + 1

            x = []
            for i in range(0, batch_size):
                tokens_group = vectorized_tokens[i:i + num_related_tokens]  # dim: [num_related_tokens, token_dim]
                x.append(tf.reshape(tokens_group, [token_dim * num_related_tokens]))

            x = tf.stack(x)  # dim: [batch_size, token_dim * num_related_tokens]

            # build the layers
            io_dims = {
                'i': token_dim * num_related_tokens,  # number of units in the input layer
                'o': num_classes,  # number of units in the output layer
            }

            layer_output = x
            for i in range(0, len(layers) - 1):
                # get dimensions for the next layer
                layer_dim_1 = utils.eval_expr(layers[i], io_dims)
                layer_dim_2 = utils.eval_expr(layers[i + 1], io_dims)

                # get the layer output
                w = tf.Variable(tf.random_normal([layer_dim_1, layer_dim_2]))
                b = tf.Variable(tf.random_normal([layer_dim_2]))
                layer_output = tf.matmul(layer_output, w) + b
                if i != len(layers) - 2:
                    layer_output = tf.nn.tanh(layer_output)

            self._raw_predictions = layer_output  # dim: [batch_size, num_classes]

            # predictions
            self._predictions = tf.argmax(self._raw_predictions, 1)  # dim: [batch_size]

    def get_predictions(self, sess, data):
        feed_dict = {self._tokens: data}
        return sess.run(self._predictions, feed_dict=feed_dict)
