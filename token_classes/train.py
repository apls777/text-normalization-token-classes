import numpy as np
import token_classes.data_sets as ds
from token_classes import session_config
from token_classes.session import Session
from token_classes.training_model import TrainingModel
import logging

np.set_printoptions(suppress=True)  # , threshold=np.nan)
logging.basicConfig(level=logging.DEBUG)

restore_latest_session = False
restore_session_id = 0

# session directory
session_dir = session_config.get_session_dir(restore_latest_session, restore_session_id)

# trying to read a model configuration
config = session_config.read_config(session_dir)
if not config:
    # create a new model configuration
    config = {
        'params': {
            'num_chars': 15,  # maximum number of character in a token
            'token_dim': 100,  # token dimensionality after LSTM
            'num_token_dependencies': 2,  # token class depends on X tokens to the left
            'token_num_layers': 1,  # number of LSTM layers for tokens
        },
        'files': {
            'chars_groups': '../data/train/chars_groups.txt',
            'token_classes': '../data/train/token_classes.txt',
        },
    }

    # save a model configuration to the session directory
    session_config.write_config(session_dir, config)

# training parameters
batch_size = 100
learning_rate = 0.0001
checkpoint_steps = 500  # how often to save a model and check an accuracy
tokens_limit = 50000
all_tokens_file = '../data/train/all_tokens.csv'

# read the training data
data_sets, char_dim, num_classes = ds.read_data(all_tokens_file, config['files']['chars_groups'],
                                                config['files']['token_classes'], config['params']['num_chars'],
                                                tokens_limit)

# create model
model = TrainingModel(learning_rate, char_dim, config['params']['token_dim'], config['params']['num_chars'],
                      num_classes, batch_size, config['params']['num_token_dependencies'],
                      config['params']['token_num_layers'])

# create session
session = Session(model, session_dir)

# run training
session.train(data_sets, batch_size, checkpoint_steps)
