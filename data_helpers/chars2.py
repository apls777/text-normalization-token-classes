import csv

all_tokens_file = '../data/train/all_tokens.csv'
chars_file = '../data/analysis/chars2.csv'

with open(all_tokens_file, mode='r', encoding='utf8') as af:
    reader = csv.reader(af)
    titles = next(reader)

    chars = {}

    counter = 0
    for (sentence_id, token_id, class_name, str_before, str_after) in reader:
        sentence_id = int(sentence_id)
        token_id = int(token_id)

        counter += 1
        if counter % 100000 == 0:
            print('  processing data row %d' % counter)

        # skip strings with "sil", we don't know which one character is "sil"
        if 'sil' in str_after:
            continue

        # skip unchanged single characters or strings
        if len(str_before) == 1 and str_before == str_after:
            continue

        for char in str_before:
            if char not in chars:
                chars[char] = 1
            else:
                chars[char] += 1

    chars = sorted(list(chars.items()), key=lambda x: x[1], reverse=True)

print('Writing the file')

with open(chars_file, mode='w', encoding='utf8') as nf:
    writer = csv.writer(nf, quoting=csv.QUOTE_NONNUMERIC, lineterminator='\n')
    writer.writerow(('char', 'count'))
    writer.writerows(chars)
