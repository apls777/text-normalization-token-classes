import csv
import logging
from data_set import DataSet
from utils import DataSets


def read_data(all_tokens_file: str, chars_groups_file: str, token_classes_file: str,
              num_chars: int, limit: int = 0, offset: int = 0) -> (DataSets, int, int):
    logger = logging.getLogger(__name__)

    # read chars groups
    chars_dict, char_dim = read_chars_groups(chars_groups_file)

    # read token classes
    classes_dict, num_classes = read_token_classes(token_classes_file)

    logger.debug('Reading the tokens file: "%s"' % all_tokens_file)
    with open(all_tokens_file, mode='r', encoding='utf8') as f:
        reader = csv.reader(f)
        next(reader)

        counter = 0

        cur_sentence_id, token_id, class_name, str_before, str_after = next(reader)

        cur_sentence_id = int(cur_sentence_id)
        token_id = int(token_id)
        chars_ids = get_chars_ids(str_before, chars_dict, num_chars)  # dim: [num_chars]
        class_id = get_class_id(class_name, classes_dict)

        # pad sentence with two empty tokens
        tokens = [[-1] * num_chars, [-1] * num_chars, chars_ids]  # dim: [num_tokens, num_chars]
        classes_ids = [0, 0, class_id]  # dim: [num_tokens]
        tokens_keys = [(-1, -1), (-1, -1), (cur_sentence_id, token_id)]  # dim: [num_tokens]

        for (sentence_id, token_id, class_name, str_before, str_after) in reader:
            counter += 1
            if counter % 100000 == 0:
                logger.debug('Processing data row %d' % counter)

            if counter <= offset:
                continue

            if limit and counter > limit:
                break

            sentence_id = int(sentence_id)
            token_id = int(token_id)
            chars_ids = get_chars_ids(str_before, chars_dict, num_chars)  # dim: [num_chars]
            class_id = get_class_id(class_name, classes_dict)

            if sentence_id == cur_sentence_id:
                tokens.append(chars_ids)
                classes_ids.append(class_id)
                tokens_keys.append((sentence_id, token_id))
            else:
                cur_sentence_id = sentence_id
                # pad sentence with two empty tokens
                tokens += [[-1] * num_chars, [-1] * num_chars, chars_ids]
                classes_ids += [0, 0, class_id]
                tokens_keys += [(-1, -1), (-1, -1), (sentence_id, token_id)]

    validation_size = min(int(len(tokens) * 0.1), 10000)
    test_size = 0  # int(len(data) * 0.2)
    train_size = len(tokens) - validation_size - test_size

    logger.debug('Train size=%d, Validation size=%d, Test size=%d' % (train_size, validation_size, test_size))

    train_data = tokens[:train_size]
    train_labels = classes_ids[:train_size]
    train_keys = tokens_keys[:train_size]
    validation_data = tokens[train_size:train_size + validation_size]
    validation_labels = classes_ids[train_size:train_size + validation_size]
    validation_keys = tokens_keys[train_size:train_size + validation_size]
    test_data = tokens[train_size + validation_size:]
    test_labels = classes_ids[train_size + validation_size:]
    test_keys = tokens_keys[train_size + validation_size:]

    train = DataSet(train_data, train_labels, train_keys)
    validation = DataSet(validation_data, validation_labels, validation_keys)
    test = DataSet(test_data, test_labels, test_keys)

    return DataSets(train=train, validation=validation, test=test), char_dim, num_classes


def get_chars_ids(token: str, chars_dict: dict, num_chars: int):
    token = token[:num_chars]
    chars_ids = []
    for char in token:
        chars_ids.append(chars_dict[char] if char in chars_dict else chars_dict['UNK'])

    chars_ids += [-1] * (num_chars - len(token))

    return chars_ids


def get_class_id(class_name: str, classes_dict: dict) -> int:
    if class_name not in classes_dict:
        raise ValueError('Class name "%s" not found' % class_name)

    return classes_dict[class_name]


def read_chars_groups(chars_groups_file: str) -> (dict, int):
    logger = logging.getLogger(__name__)
    logger.debug('Reading the char groups: "%s"' % chars_groups_file)

    with open(chars_groups_file, mode='r', encoding='utf8') as f:
        chars_groups = f.readlines()
        chars_groups = [x.strip('\n') for x in chars_groups]
        char_dim = len(chars_groups) + 1  # plus one for unknown characters
        chars_dict = {'UNK': 0}
        for i, chars_group in enumerate(chars_groups):
            for char in chars_group:
                if char in chars_dict:
                    raise ValueError('Duplicated character')
                chars_dict[char] = i + 1

        logger.debug('%d char groups were read' % len(chars_groups))

    return chars_dict, char_dim


def read_token_classes(token_classes_file: str) -> (dict, int):
    logger = logging.getLogger(__name__)
    logger.debug('Reading the token classes: "%s"' % token_classes_file)

    with open(token_classes_file, mode='r', encoding='utf8') as f:
        token_classes = f.readlines()
        token_classes = [x.strip('\n') for x in token_classes]
        num_classes = len(token_classes)

        classes_dict = {}
        for i, class_name in enumerate(token_classes):
            classes_dict[class_name] = i

        logger.debug('%d token classes were read' % num_classes)

    return classes_dict, num_classes
