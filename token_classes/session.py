import os
import tensorflow as tf
import utils
from data_set import DataSet
from token_classes.data_sets import DataSets
from token_classes.model import Model
import logging


class Session(object):
    def __init__(self, model: Model, training_dir: str, restore_latest_session: bool = False):
        self._logger = logging.getLogger(__name__)
        self._model = model

        # create a session
        self._sess = tf.Session(graph=self._model.graph)

        # get ID of last saved session
        dirs = sorted(os.listdir(training_dir), reverse=True)
        last_session_id = int(dirs[0].split('_')[1]) if dirs else 0

        # set ID for current session
        if restore_latest_session and last_session_id:
            session_id = last_session_id
        else:
            session_id = last_session_id + 1

        # paths to session files and to tensorboard logs
        self._session_path = os.path.join(training_dir, 'session_%d' % session_id)
        self._logs_path = os.path.join(self._session_path, 'logs')

        # session saver
        with self._sess.graph.as_default():
            self._saver = tf.train.Saver()

        # restore a checkpoint or initialize a model
        checkpoint = tf.train.get_checkpoint_state(self._session_path)
        if checkpoint and tf.train.checkpoint_exists(checkpoint.model_checkpoint_path):
            self._logger.info('Initializing model with parameters from "%s"' % checkpoint.model_checkpoint_path)
            self._saver.restore(self._sess, checkpoint.model_checkpoint_path)
        else:
            self._logger.info('Initializing model with fresh parameters')
            utils.check_path(self._session_path)
            self._model.init_variables(self._sess)

    def train(self, data_sets: DataSets, batch_size: int, checkpoint_steps: int):
        # init summary writer
        utils.check_path(self._logs_path)
        summary_writer = tf.summary.FileWriter(self._logs_path, self._sess.graph)

        self._logger.debug('Start training')

        # start training
        while True:
            # get batch of data
            data, labels = data_sets.train.next_batch(batch_size)

            # perform training step
            summary_val, global_step = self._model.train_step(self._sess, data, labels)
            summary_writer.add_summary(summary_val, global_step)

            if global_step % checkpoint_steps == 0:
                # save session
                save_path = self._save_session(global_step)
                self._logger.info('Session was saved in the file: %s' % save_path)

                # display current loss and accuracy
                validation_steps = 10
                cost = accuracy = 0
                for i in range(0, validation_steps):
                    data, labels = data_sets.validation.next_batch(batch_size)
                    res = self._model.evaluate_model(self._sess, data, labels)
                    cost += res[0]
                    accuracy += res[1]

                cost /= validation_steps
                accuracy /= validation_steps

                self._logger.info('Step=%d, Loss=%.5f, Accuracy=%.5f' % (global_step, cost, accuracy))

    def predict(self, data_set: DataSet):
        return self._model.get_predictions(self._sess, data_set)

    def _save_session(self, global_step):
        checkpoint_path = os.path.join(self._session_path, 'model.ckpt')
        return self._saver.save(self._sess, checkpoint_path, global_step=global_step)

    def __del__(self):
        self._sess.close()
