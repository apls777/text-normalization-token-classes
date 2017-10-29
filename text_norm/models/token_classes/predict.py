from text_norm.models.token_classes.session import Session
from text_norm.utils import root_dir


# create session
session_id = 'session_13'
session = Session(__file__, session_id)

# prediction parameters
all_tokens_file = root_dir('data/train/all_tokens.csv')
batch_size = 100
tokens_limit = 0
tokens_offset = 0

# run predicting
session.collect_errors(all_tokens_file, batch_size, tokens_limit, tokens_offset)
