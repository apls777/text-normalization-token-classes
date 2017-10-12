import json
import logging
import os
from shutil import copyfile
import numpy as np
import tensorflow as tf
import token_classes.data_sets as ds
import utils
from abstract_model import AbstractModel
from token_classes.helpers import write_errors_file
from token_classes.prediction_model import PredictionModel
from token_classes.training_model import TrainingModel


class Session(object):
    def __init__(self, latest_session: bool = False, session_id: int = 0, default_config: dict = None):
        self._logger = logging.getLogger(__name__)

        # session directory
        self._session_dir = self._get_session_dir(latest_session, session_id)

        # paths to checkpoints and tensorboard logs
        self._checkpoints_dir = os.path.join(self._session_dir, 'checkpoints')
        self._logs_dir = os.path.join(self._session_dir, 'logs')

        # load configuration
        self._config = self._read_config(self._session_dir)
        if not self._config:
            if default_config is not None:
                # create a new model configuration
                self._write_config(self._session_dir, default_config)
                self._config = default_config
            else:
                raise ValueError('Config not loaded')

    @property
    def session_dir(self):
        return self._session_dir

    def train(self, all_tokens_file: str, batch_size: int, learning_rate: float, checkpoint_steps: int,
              tokens_limit: int = 0):
        config = self._config

        # read the training data
        data_sets, char_dim, num_classes = ds.read_data(all_tokens_file, config['files']['chars_groups'],
                                                        config['files']['token_classes'], config['params']['num_chars'],
                                                        tokens_limit)

        # create a training model
        model = TrainingModel(learning_rate, char_dim, config['params']['token_dim'], config['params']['num_chars'],
                              num_classes, batch_size, config['params']['num_tokens_left'],
                              config['params']['num_tokens_right'], config['params']['token_num_layers'],
                              config['params']['layers'])

        # get a session
        sess = self._init_session(model)

        # init summary writer
        utils.check_path(self._logs_dir)
        summary_writer = tf.summary.FileWriter(self._logs_dir, sess.graph)

        self._logger.debug('Start training')

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

    def collect_errors(self, tokens_file: str, errors_file: str, batch_size: int, limit: int = 0, offset: int = 0):
        config = self._config

        # read the training data
        data_sets, char_dim, num_classes = ds.read_data(tokens_file, config['files']['chars_groups'],
                                                        config['files']['token_classes'], config['params']['num_chars'],
                                                        limit, offset)

        # create a prediction model
        model = PredictionModel(char_dim, config['params']['token_dim'], config['params']['num_chars'],
                                num_classes, batch_size, config['params']['num_tokens_left'],
                                config['params']['num_tokens_right'], config['params']['token_num_layers'],
                                config['params']['token_num_layers'])

        # get a session
        sess = self._init_session(model)

        tokens_errors = {}
        classes_missed = [0] * num_classes
        classes_wrong = [0] * num_classes
        accuracy = 0
        c = 0

        self._logger.debug('Start predicting')

        while True:
            # get batch of data
            data, labels, keys = data_sets.train.next_batch(batch_size)
            if data_sets.train.epoch == 2:
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

        classes_dict, num_classes = ds.read_token_classes(config['files']['token_classes'])
        write_errors_file(tokens_file, errors_file, tokens_errors, classes_missed, classes_wrong, classes_dict, limit,
                          offset)
        sess.close()

    def _get_session_dir(self, latest_session: bool = False, session_id: int = 0):
        training_dir = os.path.join('..', 'training')
        latest_session_id = self._get_latest_session_id(training_dir, 'session')

        if latest_session:
            session_id = latest_session_id

        if not session_id:
            session_id = latest_session_id + 1

        # session directory
        session_dir = os.path.join(training_dir, 'session_%d' % session_id)

        return session_dir

    def _get_latest_session_id(self, training_dir, prefix):
        """
        Get an ID of the last saved configuration
        """
        prefix += '_'
        latest_session_id = 0

        dirs = os.listdir(training_dir)
        for dir in dirs:
            if dir.startswith(prefix):
                session_id = int(dir[len(prefix):])
                if session_id > latest_session_id:
                    latest_session_id = session_id

        return latest_session_id

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
        for key, filename in config['files'].items():
            new_filename = os.path.join(files_dir, os.path.basename(filename))
            copyfile(filename, new_filename)
            config['files'][key] = new_filename
            self._logger.debug('File "%s" was copied' % filename)

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
