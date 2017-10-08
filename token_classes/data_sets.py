import csv
import logging
import numpy as np
from data_set import DataSet
from utils import DataSets


def read_data(all_tokens_file: str, chars_groups_file: str, token_classes_file: str,
              num_chars: int, num_words: int, limit: int = 0, offset: int = 0) -> (DataSets, int, int):
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

    logger.debug('Reading the token classes: "%s"' % token_classes_file)
    with open(token_classes_file, mode='r', encoding='utf8') as f:
        token_classes = f.readlines()
        token_classes = [x.strip('\n') for x in token_classes]
        num_classes = len(token_classes)

        classes_dict = {}
        for i, class_name in enumerate(token_classes):
            classes_dict[class_name] = i

        logger.debug('%d token classes were read' % num_classes)

    logger.debug('Reading the tokens file: "%s"' % all_tokens_file)
    with open(all_tokens_file, mode='r', encoding='utf8') as f:
        reader = csv.reader(f)
        next(reader)

        counter = 0

        cur_sentence_id, token_id, class_name, str_before, str_after = next(reader)
        cur_sentence_id = int(cur_sentence_id)
        vectorized_token = get_vectorized_token(str_before, chars_dict, num_chars,
                                                char_dim)  # dim: [num_chars, char_dim]

        sentences = [[vectorized_token]]  # dim: [num_sentences, num_words, num_chars, char_dim]
        labels_ids = [[0]]  # dim: [num_sentences, num_words]

        for (sentence_id, token_id, class_name, str_before, str_after) in reader:
            sentence_id = int(sentence_id)
            vectorized_token = get_vectorized_token(str_before, chars_dict, num_chars, char_dim)

            counter += 1
            if counter % 100000 == 0:
                logger.debug('Processing data row %d' % counter)

            if counter <= offset:
                continue

            if limit and counter > limit:
                break

            if class_name not in classes_dict:
                raise ValueError('Class name "%s" not found' % class_name)

            label_id = classes_dict[class_name]

            if sentence_id == cur_sentence_id:
                sentences[len(sentences) - 1].append(vectorized_token)
                labels_ids[len(labels_ids) - 1].append(label_id)
            else:
                cur_sentence_id = sentence_id
                sentences.append([vectorized_token])
                labels_ids.append([label_id])

    logger.debug('Fixing the shape')
    labels = []  # dim: [num_sentences, num_words, num_classes]
    for i, sentence in enumerate(sentences):
        sentences[i] = sentence[:num_words]
        labels_ids[i] = labels_ids[i][:num_words]
        for j in range(num_words - len(sentence)):
            sentences[i].append(get_vectorized_token('', chars_dict, num_chars, char_dim))
            labels_ids[i].append(0)

        labels.append(get_one_hot_matrix(num_words, num_classes, labels_ids[i]))

    # dim: [(num_sentences * num_words), num_chars, char_dim]
    data = np.reshape(np.array(sentences), (num_words * len(sentences), num_chars, -1))

    # dim: [(num_sentences * num_words), num_classes]
    labels = np.reshape(np.array(labels), (num_words * len(labels), -1))

    validation_size = min(int(len(data) * 0.1), 1000)
    test_size = 0  # int(len(data) * 0.2)
    train_size = len(data) - validation_size - test_size

    logger.debug('Train size=%d, Validation size=%d, Test size=%d' % (train_size, validation_size, test_size))

    train_data = data[:train_size]
    train_labels = labels[:train_size]
    validation_data = data[train_size:train_size + validation_size]
    validation_labels = labels[train_size:train_size + validation_size]
    test_data = data[train_size + validation_size:]
    test_labels = labels[train_size + validation_size:]

    train = DataSet(train_data, train_labels)
    validation = DataSet(validation_data, validation_labels)
    test = DataSet(test_data, test_labels)

    return DataSets(train=train, validation=validation, test=test), char_dim, num_classes


def get_vectorized_token(token: str, chars_dict: dict, num_chars: int, char_dim: int):
    token = token[:num_chars]
    chars_ids = []
    for char in token:
        chars_ids.append(chars_dict[char] if char in chars_dict else chars_dict['UNK'])

    # one hot vectors
    return get_one_hot_matrix(num_chars, char_dim, chars_ids)


def get_one_hot_matrix(num_vectors: int, vector_dim: int, ids: list) -> np.ndarray:
    one_hot_matrix = np.zeros((num_vectors, vector_dim))
    one_hot_matrix[np.arange(len(ids)), ids] = 1

    return one_hot_matrix
