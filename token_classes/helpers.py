import csv
import logging


def write_errors_file(tokens_file: str, output_file: str, token_errors: dict, classes_missed: list,
                      classes_wrong: list, classes_dict: dict, limit: int = 0, offset: int = 0,
                      logger: logging.Logger = None):
    if logger is None:
        logger = logging.getLogger(__name__)

    with open(output_file, mode='w', encoding='utf8') as of:
        classes_names = [''] * len(classes_dict)
        for class_name, label_id in classes_dict.items():
            of.write(class_name + ': ' + str(classes_missed[label_id]) + ' missed, ' + str(
                classes_wrong[label_id]) + ' wrong\n')
            classes_names[label_id] = class_name
        of.write('\n')

        with open(tokens_file, mode='r', encoding='utf8') as tf:
            reader = csv.reader(tf)

            next(reader)

            counter = 0
            sentence_before = ''
            cur_sentence_id = None
            has_error = False

            for (sentence_id, token_id, class_name, str_before, str_after) in reader:
                sentence_id = int(sentence_id)
                token_id = int(token_id)

                counter += 1
                if counter % 100000 == 0:
                    logger.debug('Processing data row %d' % counter)

                if counter <= offset:
                    continue

                if limit and counter > limit:
                    break

                if sentence_id == cur_sentence_id:
                    sentence_before += ' ' + str_before
                else:
                    if has_error:
                        of.write(sentence_before + '\n\n')
                    cur_sentence_id = sentence_id
                    sentence_before = str_before
                    has_error = False

                if sentence_id in token_errors and token_id in token_errors[sentence_id]:
                    has_error = True
                    truth_id, predicted_id = token_errors[sentence_id][token_id]
                    of.write('> ' + str_before + ' (truth: ' + classes_names[truth_id] + ', predicted: ' +
                             classes_names[predicted_id] + ')\n')
