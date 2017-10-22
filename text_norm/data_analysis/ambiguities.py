import csv

all_tokens_file = '../data/train/all_tokens.csv'
ambiguities_file = '../data/analysis/ambiguities.csv'

with open(ambiguities_file, mode='w', encoding='utf8') as nf:
    with open(all_tokens_file, mode='r', encoding='utf8') as af:
        reader = csv.reader(af)
        writer = csv.writer(nf, quoting=csv.QUOTE_NONNUMERIC, lineterminator='\n')

        titles = next(reader)
        writer.writerow(titles)

        strs = {}
        counter = 0
        for (sentence_id, token_id, class_name, str_before, str_after) in reader:
            sentence_id = int(sentence_id)
            token_id = int(token_id)

            counter += 1
            if counter % 100000 == 0:
                print('  processing data row %d' % counter)

            if str_before not in strs:
                strs[str_before] = {str_after: (sentence_id, token_id, class_name)}
            elif str_after not in strs[str_before]:
                strs[str_before][str_after] = (sentence_id, token_id, class_name)

        print('Writing the file')

        for str_before in strs:
            if len(strs[str_before]) > 1:
                for str_after in strs[str_before]:
                    sentence_id, token_id, class_name = strs[str_before][str_after]
                    writer.writerow((sentence_id, token_id , class_name, str_before, str_after))
