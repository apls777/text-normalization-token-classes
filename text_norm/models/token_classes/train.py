import argparse
from text_norm.models.token_classes.session import Session
from text_norm.utils import root_dir

parser = argparse.ArgumentParser()
parser.add_argument('-r', action='store_true', help='Restore the last session')
parser.add_argument('-s', type=int, default=0, help='Session ID to continue a training')

args = parser.parse_args()

restore_latest_session = args.r  # restores the last session or creates new one (if True, overrides session_id)
session_id = args.s  # use particular session ID, it restores existing session or creates new one

# default model configuration
default_config = {
    'params': {
        'num_chars': 20,  # maximum number of character in a token
        'token_dim': 300,  # token dimensionality after LSTM
        'num_tokens_left': 2,  # token class depends on X tokens to the left
        'num_tokens_right': 2,  # token class depends on X tokens to the right
        'token_num_layers': 2,  # number of LSTM layers for tokens
        'layers': ['i', 'o'],  # layers for feed forward network
    },
    'files': {
        'chars_groups': 'config/chars_groups_2.txt',
        'token_classes': 'config/token_classes.txt',
    },
}

# training parameters
all_tokens_file = root_dir('data/train/all_tokens.csv')
batch_size = 100
learning_rate = 0.0001
checkpoint_steps = 5000  # how often to save a model and check an accuracy
tokens_limit = 0

# create session
session = Session(__file__, restore_latest_session, session_id, default_config)

# run training
session.train(all_tokens_file, batch_size, learning_rate, checkpoint_steps, tokens_limit)
