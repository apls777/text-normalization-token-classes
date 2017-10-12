from token_classes.session import Session
import logging

logging.basicConfig(level=logging.DEBUG)

# create session
session_id = 10
session = Session(False, session_id)

# prediction parameters
all_tokens_file = '../data/train/all_tokens.csv'
errors_file = '../data/analysis/errors_' + str(session_id) + '.txt'
batch_size = 100
tokens_limit = 5000
tokens_offset = 0

# run predicting
session.collect_errors(all_tokens_file, errors_file, batch_size, tokens_limit, tokens_offset)
