import csv

all_tokens_file = '../data/train/all_tokens.csv'
chars_file = '../data/analysis/chars.csv'
punct_file = '../data/analysis/punct.csv'
verbatims_single_file = '../data/analysis/verbatims_single.csv'
verbatims_greek_file = '../data/analysis/verbatims_greek.csv'
verbatims_in_chars_file = '../data/analysis/verbatims_in_chars.csv'

with open(all_tokens_file, mode='r', encoding='utf8') as af:
    reader = csv.reader(af)
    titles = next(reader)

    chars = {}
    punct = {}
    verbatims_single = {}
    verbatims_greek = {}
    verbatims_in_chars = []

    counter = 0
    for (sentence_id, token_id, class_name, str_before, str_after) in reader:
        sentence_id = int(sentence_id)
        token_id = int(token_id)

        counter += 1
        if counter % 100000 == 0:
            print('  processing data row %d' % counter)

        if class_name == 'VERBATIM':
            if str_before == str_after:
                if str_before not in verbatims_single:
                    verbatims_single[str_before] = 1
                else:
                    verbatims_single[str_before] += 1
            else:
                if str_before not in verbatims_greek:
                    verbatims_greek[str_before] = 1
                else:
                    verbatims_greek[str_before] += 1
        elif class_name == 'PUNCT':
            if str_before not in punct:
                punct[str_before] = 1
            else:
                punct[str_before] += 1
        else:
            for char in list(str_before):
                if char not in chars:
                    chars[char] = 1
                else:
                    chars[char] += 1

    for verbatim in chars:
        if verbatim in verbatims_single or verbatim in verbatims_greek:
            verbatims_in_chars.append((verbatim, chars[verbatim]))

    chars = sorted(list(chars.items()), key=lambda x: x[1], reverse=True)
    punct = sorted(list(punct.items()), key=lambda x: x[1], reverse=True)
    verbatims_single = sorted(list(verbatims_single.items()), key=lambda x: x[1], reverse=True)
    verbatims_greek = sorted(list(verbatims_greek.items()), key=lambda x: x[1], reverse=True)
    verbatims_in_chars = sorted(verbatims_in_chars, key=lambda x: x[1], reverse=True)

print('Writing the files')

with open(chars_file, mode='w', encoding='utf8') as nf:
    writer = csv.writer(nf, quoting=csv.QUOTE_NONNUMERIC, lineterminator='\n')
    writer.writerow(('char', 'count'))
    writer.writerows(chars)

with open(punct_file, mode='w', encoding='utf8') as nf:
    writer = csv.writer(nf, quoting=csv.QUOTE_NONNUMERIC, lineterminator='\n')
    writer.writerow(('punct', 'count'))
    writer.writerows(punct)

with open(verbatims_single_file, mode='w', encoding='utf8') as nf:
    writer = csv.writer(nf, quoting=csv.QUOTE_NONNUMERIC, lineterminator='\n')
    writer.writerow(('verbatim', 'count'))
    writer.writerows(verbatims_single)

with open(verbatims_greek_file, mode='w', encoding='utf8') as nf:
    writer = csv.writer(nf, quoting=csv.QUOTE_NONNUMERIC, lineterminator='\n')
    writer.writerow(('verbatim', 'count'))
    writer.writerows(verbatims_greek)

with open(verbatims_in_chars_file, mode='w', encoding='utf8') as nf:
    writer = csv.writer(nf, quoting=csv.QUOTE_NONNUMERIC, lineterminator='\n')
    writer.writerow(('verbatim', 'in_chars_count'))
    writer.writerows(verbatims_in_chars)
