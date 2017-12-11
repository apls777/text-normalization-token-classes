import os
import numpy as np
import tensorflow as tf
import text_norm.models.token_classes.data_sets as ds
from text_norm.abstract_session import AbstractSession
from text_norm.models.token_classes.helpers import write_errors_file
from text_norm.models.token_classes.prediction_model import PredictionModel
from text_norm.models.token_classes.training_model import TrainingModel


class Session(AbstractSession):
    def train(self, all_tokens_file: str, batch_size: int, learning_rate: float, checkpoint_steps: int,
              tokens_limit: int = 0, tokens_offset: int = 0):
        config = self._config

        # read the training data
        chars_groups_filename = os.path.join(self._session_dir, config['files']['chars_groups'])
        token_classes_filename = os.path.join(self._session_dir, config['files']['token_classes'])
        data_sets, char_dim, num_classes = ds.read_data(all_tokens_file, chars_groups_filename,
                                                        token_classes_filename, config['params']['num_chars'],
                                                        tokens_limit, tokens_offset, self._logger)

        # create a training model
        model = TrainingModel(learning_rate, char_dim, config['params']['token_dim'], config['params']['num_chars'],
                              num_classes, batch_size, config['params']['num_tokens_left'],
                              config['params']['num_tokens_right'], config['params']['token_num_layers'],
                              config['params']['layers'], self._logger)

        # get a session
        sess = self._init_session(model)

        # init summary writer
        summary_writer = tf.summary.FileWriter(self._logs_dir, sess.graph)

        self._logger.debug('===== Start training =====')
        self._logger.debug('Batch Size=%d; Learning Rate=%f' % (batch_size, learning_rate))

        # start training
        while True:
            # get batch of data
            data, labels, keys = data_sets.train.next_batch(batch_size, True)

            # perform training step
            summary_val, global_step = model.train_step(sess, data, labels)
            summary_writer.add_summary(summary_val, global_step)

            if global_step % checkpoint_steps == 0:
                # save session
                save_path = self._save_session(sess, global_step)
                self._logger.info('Session was saved in the file: %s' % save_path)

                # display current loss and accuracy
                validation_steps = 10
                cost = accuracy = 0
                for i in range(0, validation_steps):
                    data, labels, keys = data_sets.validation.next_batch(batch_size, True)
                    res = model.evaluate_model(sess, data, labels)
                    cost += res[0]
                    accuracy += res[1]

                cost /= validation_steps
                accuracy /= validation_steps

                self._logger.info('Step=%d, Loss=%.5f, Accuracy=%.5f' % (global_step, cost, accuracy))

    def collect_errors(self, tokens_file: str, batch_size: int, limit: int = 0, offset: int = 0):
        config = self._config

        # read the training data
        chars_groups_filename = os.path.join(self._session_dir, config['files']['chars_groups'])
        token_classes_filename = os.path.join(self._session_dir, config['files']['token_classes'])
        data_sets, char_dim, num_classes = ds.read_data(tokens_file, chars_groups_filename,
                                                        token_classes_filename, config['params']['num_chars'],
                                                        limit, offset, self._logger)

        # create a prediction model
        model = PredictionModel(char_dim, config['params']['token_dim'], config['params']['num_chars'],
                                num_classes, batch_size, config['params']['num_tokens_left'],
                                config['params']['num_tokens_right'], config['params']['token_num_layers'],
                                config['params']['layers'], self._logger)

        # get a session
        sess = self._init_session(model)

        tokens_errors = {}
        classes_missed = [0] * num_classes
        classes_wrong = [0] * num_classes
        accuracy = 0
        c = 0

        self._logger.debug('===== Start predicting =====')

        while True:
            # get batch of data
            data, labels, keys = data_sets.validation.next_batch(batch_size)
            if data_sets.validation.epoch > 1:
                break

            predictions = model.get_predictions(sess, data)
            accuracy += np.mean(np.equal(predictions, labels))
            c += 1

            # collect stats
            for i, label_id in enumerate(labels):
                if label_id != predictions[i]:
                    # add error
                    sentence_id, token_id = keys[i]
                    if sentence_id not in tokens_errors:
                        tokens_errors[sentence_id] = {}

                    tokens_errors[sentence_id][token_id] = (label_id, predictions[i])

                    # increment missed and wrong classes
                    classes_missed[label_id] += 1
                    classes_wrong[predictions[i]] += 1

        self._logger.debug('Accuracy: ' + str(accuracy / c))
        self._logger.debug('Start writing the errors file')

        classes_dict, num_classes = ds.read_token_classes(token_classes_filename, self._logger)
        errors_file = os.path.join(self._logs_dir, 'errors.txt')
        write_errors_file(tokens_file, errors_file, tokens_errors, classes_missed, classes_wrong, classes_dict, limit,
                          offset, self._logger)
        sess.close()
