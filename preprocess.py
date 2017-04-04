
'''Preprocessing code for network on bAbI dataset.
References:
- Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush,
  "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks",
  http://arxiv.org/abs/1502.05698
- Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
  "End-To-End Memory Networks",
  http://arxiv.org/abs/1503.08895
'''

from functools import reduce
import tarfile
import re
import numpy as np


from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences
    that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file,
    retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data


class BabiVectorizer:
    challenges = {
        # QA1 with 10,000 samples
        'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
        # QA2 with 10,000 samples
        'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
    }
    lookup_challenge = {1:'single_supporting_fact_10k', 2: 'two_supporting_facts_10k' }
    def __init__(self, challenge_num=1):
        try:
            path = get_file('babi-tasks-v1-2.tar.gz',
                            origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
        except:
            print('Error downloading dataset, please download it manually:\n'
                  '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
                  '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
            raise
        tar = tarfile.open(path)

        challenge = self.challenges[self.lookup_challenge[challenge_num]]
        train_stories = get_stories(tar.extractfile(challenge.format('train')))
        test_stories = get_stories(tar.extractfile(challenge.format('test')))

        vocab = set()
        for story, q, answer in train_stories + test_stories:
            vocab |= set(story + q + [answer])
        vocab = sorted(vocab)

        vocab_size = len(vocab) + 1
        story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
        query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

        word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
        idx_word = {value: key for (key, value) in word_idx.items()}  # reverse lookup
        idx_word.update({0: '~'})

        self._vocab = vocab
        self._vocab_size = vocab_size
        self._word_idx = word_idx
        self._idx_word = idx_word
        self.story_maxlen = story_maxlen
        self.query_maxlen = query_maxlen
        self._train_stories = train_stories
        self._test_stories = test_stories


    def vectorize(self, datatype='train'):
        if datatype == 'train':
            data = self.train_stories
        elif datatype == 'test':
            data = self.test_stories
        else:
            raise ValueError('Invalid argument "datatype" specified: {}'.format(datatype))

        X = []
        Xq = []
        Y = []
        for story, query, answer in data:
            x = [self.word_idx[w] for w in story]
            xq = [self.word_idx[w] for w in query]
            # let's not forget that index 0 is reserved
            y = np.zeros(len(self.word_idx) + 1)
            y[self.word_idx[answer]] = 1
            X.append(x)
            Xq.append(xq)
            Y.append(y)
        return (pad_sequences(X, maxlen=self.story_maxlen),
                pad_sequences(Xq, maxlen=self.query_maxlen), np.array(Y))

    @property
    def vocab(self): return self._vocab

    @property
    def vocab_size(self): return self._vocab_size

    @property
    def word_idx(self): return self._word_idx

    @property
    def idx_word(self): return self._idx_word

    @property
    def train_stories(self): return self._train_stories

    @property
    def test_stories(self): return self._test_stories



def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return (pad_sequences(X, maxlen=story_maxlen),
            pad_sequences(Xq, maxlen=query_maxlen), np.array(Y))