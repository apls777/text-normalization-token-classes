import csv

all_tokens_file = '../data/train/all_tokens.csv'
classes_dir = '../data/train/classes/'

with open(all_tokens_file, mode='r', encoding='utf8') as af:
    reader = csv.reader(af)
    titles = next(reader)

    tokens_by_classes = {}

    counter = 0
    for (sentence_id, token_id, class_name, str_before, str_after) in reader:
        sentence_id = int(sentence_id)
        token_id = int(token_id)

        counter += 1
        if counter % 100000 == 0:
            print('  processing data row %d' % counter)

        if class_name not in tokens_by_classes:
            tokens_by_classes[class_name] = []

        tokens_by_classes[class_name].append((sentence_id, token_id, class_name, str_before, str_after))

for class_name in tokens_by_classes:
    print('Writing the ' + class_name + ' class file')
    class_file = classes_dir + class_name + '.csv'
    with open(class_file, mode='w', encoding='utf8') as nf:
        writer = csv.writer(nf, quoting=csv.QUOTE_NONNUMERIC, lineterminator='\n')
        writer.writerow(titles)
        writer.writerows(tokens_by_classes[class_name])
