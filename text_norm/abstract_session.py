import json
import logging
import os
import sys
from abc import ABC
from datetime import datetime
from shutil import copyfile
import tensorflow as tf
from text_norm.abstract_model import AbstractModel
from text_norm import utils
from text_norm.utils import root_dir


class AbstractSession(ABC):
    def __init__(self, script_filename, session_id: str = None, default_config: dict = None):
        # model name
        model_name = os.path.basename(os.path.dirname(script_filename))

        # session directory
        self._session_dir = self._get_session_dir(model_name, session_id)

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
                self._write_config(script_filename, self._session_dir, default_config)
                self._config = default_config
            else:
                raise ValueError('Config not loaded')

    def _get_session_dir(self, model_name: str, session_id: str = None):
        if not session_id:
            session_id = datetime.today().strftime('%y%m%d_%H%M')

        # session directory
        session_dir = root_dir(os.path.join('training', model_name, session_id))
        utils.check_path(session_dir)

        return session_dir

    def _read_config(self, session_dir: str) -> dict:
        config_file = os.path.join(session_dir, 'config.json')

        config = {}
        if os.path.isfile(config_file):
            # read config file
            with open(config_file) as f:
                config = json.load(f)
                self._logger.info('Config file is read: "%s"' % config_file)

        return config

    def _write_config(self, script_filename, session_dir, config: dict):
        self._logger.debug('Writing a config file')

        # clone the config
        config = dict(config)

        # copy required files to the session directory
        files_dir = os.path.join(session_dir, 'files')
        utils.check_path(files_dir)
        for key, file_path in config['files'].items():
            file_path = os.path.join(os.path.dirname(script_filename), file_path)
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
