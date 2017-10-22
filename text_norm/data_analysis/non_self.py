import csv

all_tokens_file = '../data/train/all_tokens.csv'
non_self_tokens_file = '../data/analysis/non_self_tokens.csv'

with open(non_self_tokens_file, mode='w', encoding='utf8') as nf:
    with open(all_tokens_file, mode='r', encoding='utf8') as af:
        reader = csv.reader(af)
        writer = csv.writer(nf, quoting=csv.QUOTE_NONNUMERIC, lineterminator='\n')

        titles = next(reader)
        writer.writerow(titles)

        counter = 0
        for (sentence_id, token_id, class_name, str_before, str_after) in reader:
            sentence_id = int(sentence_id)
            token_id = int(token_id)

            counter += 1
            if counter % 100000 == 0:
                print('  processing data row %d' % counter)

            if str_before != str_after:
                writer.writerow((sentence_id, token_id , class_name, str_before, str_after))
