import csv

all_tokens_file = '../data/train/all_tokens.csv'
chars_file = '../data/analysis/token_lengths.csv'

with open(all_tokens_file, mode='r', encoding='utf8') as af:
    reader = csv.reader(af)
    titles = next(reader)

    lengths = {}

    counter = 0
    for (sentence_id, token_id, class_name, str_before, str_after) in reader:
        counter += 1
        if counter % 100000 == 0:
            print('  processing data row %d' % counter)

        token_length = len(str_before)
        if token_length in lengths:
            lengths[token_length] += 1
        else:
            lengths[token_length] = 1

    lengths = sorted(list(lengths.items()), key=lambda x: x[0])

print('Writing the file')

with open(chars_file, mode='w', encoding='utf8') as nf:
    writer = csv.writer(nf, quoting=csv.QUOTE_NONNUMERIC, lineterminator='\n')
    writer.writerow(('length', 'count'))
    writer.writerows(lengths)
