from token_classes.session import Session


restore_latest_session = False  # restores the last session or creates new one (if True, overrides session_id)
session_id = 0  # use particular session ID, it restores existing session or creates new one

# default model configuration
default_config = {
    'params': {
        'num_chars': 20,  # maximum number of character in a token
        'token_dim': 100,  # token dimensionality after LSTM
        'num_tokens_left': 2,  # token class depends on X tokens to the left
        'num_tokens_right': 2,  # token class depends on X tokens to the right
        'token_num_layers': 2,  # number of LSTM layers for tokens
        'layers': ['i', '(i+o)/2', 'o'],  # layers for feed forward network
    },
    'files': {
        'chars_groups': '../data/train/chars_groups_2.txt',
        'token_classes': '../data/train/token_classes.txt',
    },
}

# training parameters
all_tokens_file = '../data/train/all_tokens.csv'
batch_size = 100
learning_rate = 0.0001
checkpoint_steps = 5000  # how often to save a model and check an accuracy
tokens_limit = 0

# create session
session = Session(__file__, restore_latest_session, session_id, default_config)

# run training
session.train(all_tokens_file, batch_size, learning_rate, checkpoint_steps, tokens_limit)
