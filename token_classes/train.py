import numpy as np
import token_classes.data_sets as ds
from token_classes.session import Session
from token_classes.model import Model
import logging

np.set_printoptions(suppress=True)  # , threshold=np.nan)
logging.basicConfig(level=logging.DEBUG)

learning_rate = 0.0001
checkpoint_steps = 500  # how often to display loss function results
num_chars = 15  # 15
num_words = 10  # 10
num_sentences = 16
token_dim = 100
restore_last_session = True
tokens_limit = 5000000
tokens_offset = 0
all_tokens_file = '../data/train/all_tokens.csv'
chars_groups_file = '../data/train/chars_groups.txt'
token_classes_file = '../data/train/token_classes.txt'
training_dir = '../training'

batch_size = num_sentences * num_words

# read the training data
data_sets, char_dim, num_classes = ds.read_data(all_tokens_file, chars_groups_file, token_classes_file,
                                                num_chars, num_words, tokens_limit, tokens_offset)

# create model
model = Model(learning_rate, char_dim, token_dim,
              num_chars, num_words, num_sentences, num_classes)

# create session
session = Session(model, training_dir, restore_last_session)

# run training
session.train(data_sets, batch_size, checkpoint_steps)
