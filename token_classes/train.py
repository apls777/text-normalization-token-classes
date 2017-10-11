import numpy as np
from token_classes.session import Session
import logging

np.set_printoptions(suppress=True)  # , threshold=np.nan)
logging.basicConfig(level=logging.DEBUG)

restore_latest_session = False
restore_session_id = 0

# default model configuration
default_config = {
    'params': {
        'num_chars': 20,  # maximum number of character in a token
        'token_dim': 100,  # token dimensionality after LSTM
        'num_tokens_left': 2,  # token class depends on X tokens to the left
        'num_tokens_right': 2,  # token class depends on X tokens to the right
        'token_num_layers': 1,  # number of LSTM layers for tokens
    },
    'files': {
        'chars_groups': '../data/train/chars_groups_2.txt',
        'token_classes': '../data/train/token_classes.txt',
    },
}

# training parameters
all_tokens_file = '../data/train/all_tokens.csv'
batch_size = 200
learning_rate = 0.0001
checkpoint_steps = 500  # how often to save a model and check an accuracy
tokens_limit = 0

# create session
session = Session(restore_latest_session, restore_session_id, default_config)

# run training
session.train(all_tokens_file, batch_size, learning_rate, checkpoint_steps, tokens_limit)
