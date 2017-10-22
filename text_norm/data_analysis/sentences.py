import csv

all_tokens_file = '../data/train/all_tokens.csv'
sentences_file = '../data/analysis/sentences.csv'

with open(sentences_file, mode='w', encoding='utf8') as nf:
    with open(all_tokens_file, mode='r', encoding='utf8') as af:
        reader = csv.reader(af)
        writer = csv.writer(nf, quoting=csv.QUOTE_NONNUMERIC, lineterminator='\n')

        next(reader)
        writer.writerow(('sentence_id', 'before', 'after'))

        counter = 0
        cur_sentence_id, token_id, class_name, sentence_before, sentence_after = next(reader)
        cur_sentence_id = int(cur_sentence_id)

        for (sentence_id, token_id, class_name, str_before, str_after) in reader:
            sentence_id = int(sentence_id)

            counter += 1
            if counter % 100000 == 0:
                print('  processing data row %d' % counter)

            if sentence_id == cur_sentence_id:
                sentence_before += ' ' + str_before
                sentence_after += ' ' + str_after
            else:
                writer.writerow((cur_sentence_id, sentence_before, sentence_after))
                cur_sentence_id = sentence_id
                sentence_before = str_before
                sentence_after = str_after

        writer.writerow((cur_sentence_id, sentence_before, sentence_after))
