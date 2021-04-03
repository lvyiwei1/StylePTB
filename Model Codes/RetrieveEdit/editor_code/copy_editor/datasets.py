import codecs
import csv
import random
from collections import namedtuple

from os.path import join

from nltk import word_tokenize, defaultdict

from editor_code.copy_editor import data
from gtd.chrono import verboserate
from gtd.io import num_lines


def create_splits(examples, train_portion, random_seed):
    """Randomly shuffle examples, then split into train/valid/test.
    
    Valid and test are equal sized.
    
    Args:
        examples (list)
        train_portion (float)
        random_seed (int)

    Returns:
        splits (dict[str, list])
    """
    # shuffle
    examples = list(examples)  # make copy before shuffling
    random.seed(random_seed)
    random.shuffle(examples)

    n = len(examples)
    n_tr = int(n * train_portion)
    n_va = (n - n_tr) / 2

    splits = {
        'train': examples[:n_tr],
        'valid': examples[n_tr:n_tr + n_va],
        'test': examples[n_tr + n_va:],
    }

    return splits


class Example(namedtuple('Example', ['question', 'answer_span', 'rule_answer', 'gold_answer', 'uid', 'valid'])):
    def __new__(cls, question, answer_span, rule_answer, gold_answer, uid, valid):
        lower = lambda seq: [w.lower() for w in seq]
        self = super(Example, cls).__new__(cls, lower(question), lower(answer_span), lower(rule_answer), lower(gold_answer), uid, valid)
        return self

    def __unicode__(self):
        fmt = lambda seq: ' '.join(seq)

        return u'Q: {q} ({a})\nR: {r}\nG: {g}\n{uid}{valid}'.format(
            q=fmt(self.question), a=fmt(self.answer_span), r=fmt(self.rule_answer), g=fmt(self.gold_answer),
            uid=self.uid, valid='' if self.valid else ' (invalid)'
        )

    def __repr__(self):
        return unicode(self).encode('utf-8')


def unicode_csv_reader(unicode_csv_data, dialect=csv.excel, **kwargs):
    # csv.py doesn't do Unicode; encode temporarily as UTF-8:
    csv_reader = csv.reader(utf_8_encoder(unicode_csv_data), dialect=dialect, **kwargs)
    for row in csv_reader:
        # decode UTF-8 back to Unicode, cell by cell:
        yield [unicode(cell, 'utf-8') for cell in row]


def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode('utf-8')


def file_rows(path, limit):
    with codecs.open(path, encoding='utf-8') as f:
        reader = unicode_csv_reader(f, delimiter='\t')
        for i, row in verboserate(enumerate(reader), total=num_lines(path)):
            if i == 0:
                continue  # skip header
            if i > limit:
                break
            yield row


class Examples(list):
    def __init__(self, limit=float('inf')):
        examples = []
        path = join(data.workspace.root, 'all_data_postedits.csv')
        skipped = 0
        for row in file_rows(path, limit):
            aspan, uid, q, r, g, ug = row
            if not (q and r and g):
                skipped += 1
                continue
            aspan = word_tokenize(aspan.lower())
            q = word_tokenize(q.lower())
            r = word_tokenize(r.lower())
            g = word_tokenize(g.lower())
            valid = (ug != 'on')
            examples.append(Example(q, aspan, r, g, uid, valid))

        if skipped:
            print 'Skipped some poorly formatted examples: {}'.format(skipped)

        super(Examples, self).__init__(examples)


class RuleCandidates(defaultdict):
    def __init__(self, limit=float('inf')):
        super(RuleCandidates, self).__init__(list)
        path = join(data.workspace.root, 'augmented_postedits.csv')
        for row in file_rows(path, limit):
            uid, r, g = row
            r = r.lower().split()
            g = g.lower().split()
            self[uid].append((r, g))