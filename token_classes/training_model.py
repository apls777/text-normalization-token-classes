import tensorflow as tf

from token_classes.prediction_model import PredictionModel


class TrainingModel(PredictionModel):
    def __init__(self, learning_rate: float, char_dim: int, token_dim: int, num_chars: int, num_classes: int,
                 batch_size: int, num_token_dependencies: int, token_num_layers: int = 1):
        super().__init__(char_dim, token_dim, num_chars, num_classes, batch_size, num_token_dependencies,
                         token_num_layers)

        # modifying the model for training
        with self._graph.as_default():
            self._labels = tf.placeholder(tf.uint8, shape=(batch_size,), name='token_classes')
            self._global_step = tf.Variable(0, trainable=False)

            # convert indices to one-hot vectors
            one_hot_labels = tf.one_hot(self._labels, num_classes, axis=-1,
                                        name='classes_one_hot')  # dim: [batch_size, num_classes]

            # cross entropy
            self._cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=self._raw_predictions))

            # accuracy
            correct_prediction = tf.equal(self._predictions, tf.argmax(one_hot_labels, 1))
            self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self._train = optimizer.minimize(self._cost, global_step=self._global_step)

            # summaries
            tf.summary.scalar('loss', self._cost)
            tf.summary.scalar('accuracy', self._accuracy)
            self._summary = tf.summary.merge_all()

    def train_step(self, sess: tf.Session, data, labels):
        feed_dict = {self._tokens: data, self._labels: labels}
        summary_val, _, global_step = sess.run([self._summary, self._train, self._global_step], feed_dict=feed_dict)
        return summary_val, global_step

    def evaluate_model(self, sess: tf.Session, data, labels):
        feed_dict = {self._tokens: data, self._labels: labels}
        return sess.run([self._cost, self._accuracy], feed_dict=feed_dict)