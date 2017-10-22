import json
import logging
import os
import sys
from shutil import copyfile
import numpy as np
import tensorflow as tf
import text_norm.models.token_classes.data_sets as ds
from text_norm.abstract_model import AbstractModel
from text_norm import utils
from text_norm.models.token_classes.helpers import write_errors_file
from text_norm.models.token_classes.prediction_model import PredictionModel
from text_norm.models.token_classes.training_model import TrainingModel
from text_norm.utils import root_dir


class Session(object):
    def __init__(self, script_filename, use_last_session: bool = False, session_id: int = 0, default_config: dict = None):
        # session directory
        self._session_dir = self._get_session_dir(use_last_session, session_id)

        # paths to checkpoints and tensorboard logs
        self._checkpoints_dir = os.path.join(self._session_dir, 'checkpoints')
        self._logs_dir = os.path.join(self._session_dir, 'logs')
        utils.check_path(self._logs_dir)

        # logger configuration
        self._logger = self._get_logger(script_filename)

        # load a model configuration
        self._config = self._read_config(self._session_dir)
        if not self._config:
            if default_config is not None:
                # create a new model configuration
                self._write_config(self._session_dir, default_config)
                self._config = default_config
            else:
                raise ValueError('Config not loaded')

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

    def _get_session_dir(self, use_last_session: bool = False, session_id: int = 0):
        training_dir = root_dir(os.path.join('training', os.path.basename(os.path.dirname(__file__))))
        last_session_id = self._get_last_session_id(training_dir)

        if session_id > last_session_id:
            raise ValueError('Session ID can\'t be higher that the last session ID')

        if use_last_session:
            session_id = last_session_id

        if not session_id:
            session_id = last_session_id + 1

            # update the last session ID
            with open(os.path.join(training_dir, 'last_session'), mode='w') as f:
                f.write(str(session_id))

        # session directory
        session_dir = os.path.join(training_dir, 'session_%d' % session_id)
        utils.check_path(session_dir)

        return session_dir

    def _get_last_session_id(self, training_dir) -> int:
        """
        Get an ID of the last started session
        """

        session_filename = os.path.join(training_dir, 'last_session')

        if not os.path.exists(session_filename):
            return 0

        with open(session_filename, mode='r') as f:
            last_session_id = int(f.readline())

        return last_session_id

    def _read_config(self, session_dir: str) -> dict:
        config_file = os.path.join(session_dir, 'config.json')

        config = {}
        if os.path.isfile(config_file):
            # read config file
            with open(config_file) as f:
                config = json.load(f)
                self._logger.info('Config file is read: "%s"' % config_file)

        return config

    def _write_config(self, session_dir, config: dict):
        self._logger.debug('Writing a config file')

        # clone the config
        config = dict(config)

        # copy required files to the session directory
        files_dir = os.path.join(session_dir, 'files')
        utils.check_path(files_dir)
        for key, file_path in config['files'].items():
            file_path = os.path.join(os.path.dirname(__file__), file_path)
            filename = os.path.basename(file_path)
            copyfile(file_path, os.path.join(files_dir, filename))
            config['files'][key] = 'files/' + filename
            self._logger.debug('File "%s" was copied' % file_path)

        # create new config file
        config_file = os.path.join(session_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
            self._logger.info('Config file is written: "%s"' % config_file)

    def _init_session(self, model: AbstractModel) -> tf.Session:
        # create a session
        sess = tf.Session(graph=model.graph)

        # restore a checkpoint or initialize the model with fresh parameters
        checkpoint = tf.train.get_checkpoint_state(self._checkpoints_dir)
        if checkpoint and tf.train.checkpoint_exists(checkpoint.model_checkpoint_path):
            self._logger.info('Initializing the model with parameters from "%s"' % checkpoint.model_checkpoint_path)
            # get a session saver for the graph
            with sess.graph.as_default():
                saver = tf.train.Saver()
            saver.restore(sess, checkpoint.model_checkpoint_path)
        else:
            self._logger.info('Initializing the model with fresh parameters')
            utils.check_path(self._checkpoints_dir)
            with sess.graph.as_default():
                sess.run(tf.global_variables_initializer())

        return sess

    def _save_session(self, sess: tf.Session, global_step):
        with sess.graph.as_default():
            saver = tf.train.Saver()
        checkpoint_path = os.path.join(self._checkpoints_dir, 'model.ckpt')

        return saver.save(sess, checkpoint_path, global_step=global_step)

    def _get_logger(self, script_filename, level: int = logging.DEBUG):
        # path to the log file
        log_filename = os.path.splitext(os.path.basename(script_filename))[0] + '.log'
        log_filename = os.path.join(self._logs_dir, log_filename)

        # formatter
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

        # output to the console
        std_handler = logging.StreamHandler(sys.stdout)
        std_handler.setFormatter(formatter)

        # output to the log file
        file_handler = logging.FileHandler(log_filename, mode='a')
        file_handler.setFormatter(formatter)

        logger = logging.getLogger('session')
        logger.addHandler(std_handler)
        logger.addHandler(file_handler)
        logger.setLevel(level)

        return logger