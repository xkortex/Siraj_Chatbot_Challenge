
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
import glob
import numpy as np
import pandas as pd



from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences


def charvectorize(word, lower=True, setsize=128):
    """
    Convert a word (sequence of characters) to a n-vector of length setsize, using one-hot encoding
    :param word: Word to vectorize
    :param lower: Render word lowercase first before vectorizing
    :param setsize: Size of character set
    :return:
    >>> charvectorize('Mary')
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
       0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])


    """
    if lower:
        word = word.lower()
    vec = np.zeros(setsize, int)
    for c in word:
        vec[ord(c)] = 1
    return vec


def dist(v1, v2):
    """
    Euclidean distance
    :param v1: Vector
    :param v2: Vector or list of vectors
    :return:
    >>> dist(0.5, 0.25)
    0.25
    >>> dist((.5, .6, .7), (.3, .3, .3))
    0.53851648071345037
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    dv = v2 - v1
    dv = dv ** 2
    dv = np.sum(dv, axis=-1)
    return dv ** 0.5


def matchnocase(word, vocab):
    """
    Match a word to a vocabulary while ignoring case
    :param word: Word to try to match
    :param vocab: Valid vocabulary
    :return:
    >>> matchnocase('mary', {'Alice', 'Bob', 'Mary'})
    'Mary'
    """
    lword = word.lower()
    listvocab = list(vocab) # this trick catches dict and set in addition to list
    lvocab = [w.lower() for w in listvocab]
    if lword in lvocab:
        return listvocab[lvocab.index(lword)]
    return None


def softmatch(word, vocab, lower=True, cutoff=2.):
    """
    Try to soft-match to catch various typos.
    :param word: Word to try to match
    :param vocab: Valid vocabulary
    :param cutoff: Maximum distance (exclusive) to return match
    :return: Corrected word
    >>> softmatch('mbry', {'Alice', 'Bob', 'Mary'})
    'Mary'
    """
    listvocab = list(vocab)
    vw = charvectorize(word)
    vecs = np.array([charvectorize(w, lower=lower) for w in listvocab])
    distances = dist(vw, vecs)
    idx = np.argmin(distances)
    confidence = distances[idx]
    if confidence < cutoff:
        return listvocab[idx]
    return None

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    """
    Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences
    that support the answer are kept.
    :param lines:
    :param only_supporting:
    :return:
    """
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
    """
    Given a file name, read the file,
    retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    :param f: file
    :param only_supporting:
    :param max_length:
    :return:
    """
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data


class BabiVectorizer:
    allow_case_insensitive = True
    allow_softmatch = False
    ignore_keyerror = True
    basedir = 'tasks_1-20_v1-2/en-10k/'
    challenge_files = glob.glob(basedir + 'qa*.txt')
    # challenges = {
    #     # QA1 with 10,000 samples
    #     'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
    #     # QA2 with 10,000 samples
    #     'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
    # }
    challenges =    {1: '{}qa1_single-supporting-fact_{}.txt',
                     2: '{}qa2_two-supporting-facts_{}.txt',
                     3: '{}qa3_three-supporting-facts_{}.txt',
                     4: '{}qa4_two-arg-relations_{}.txt',
                     5: '{}qa5_three-arg-relations_{}.txt',
                     6: '{}qa6_yes-no-questions_{}.txt',
                     7: '{}qa7_counting_{}.txt',
                     8: '{}qa8_lists-sets_{}.txt',
                     9: '{}qa9_simple-negation_{}.txt',
                     10: '{}qa10_indefinite-knowledge_{}.txt',
                     11: '{}qa11_basic-coreference_{}.txt',
                     12: '{}qa12_conjunction_{}.txt',
                     13: '{}qa13_compound-coreference_{}.txt',
                     14: '{}qa14_time-reasoning_{}.txt',
                     15: '{}qa15_basic-deduction_{}.txt',
                     16: '{}qa16_basic-induction_{}.txt',
                     17: '{}qa17_positional-reasoning_{}.txt',
                     18: '{}qa18_size-reasoning_{}.txt',
                     19: '{}qa19_path-finding_{}.txt',
                     20: '{}qa20_agents-motivations_{}.txt'}



    lookup_challenge = {1:'single_supporting_fact_10k', 2: 'two_supporting_facts_10k' }
    def __init__(self, challenge_num=1):
        """
        Word Vectorizer for for Babi Dataset. Handles loading data, parsing, converting to int index, maintaining the
        vocabulary, and converting back from vectors to sentences.
        :param challenge_num: {1|2} Specify the challenge which to load.
                1 = One supporting fact
                2 = Two supporting facts
        """
        try:
            path = get_file('babi-tasks-v1-2.tar.gz',
                            origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
        except:
            print('Error downloading dataset, please download it manually:\n'
                  '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
                  '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
            raise
        tar = tarfile.open(path)

        challenge = self.challenges[challenge_num]
        print('Loading: {}'.format(challenge))
        train_records = get_stories(tar.extractfile(challenge.format(self.basedir, 'train')))
        test_records = get_stories(tar.extractfile(challenge.format(self.basedir, 'test')))

        vocab = set()
        for story, q, answer in train_records + test_records:
            vocab |= set(story + q + [answer])
        vocab = sorted(vocab)

        vocab_size = len(vocab) + 1
        story_maxlen = max(map(len, (x for x, _, _ in train_records + test_records)))
        query_maxlen = max(map(len, (x for _, x, _ in train_records + test_records)))

        word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
        idx_word = {value: key for (key, value) in word_idx.items()}  # reverse lookup
        idx_word.update({0: ''})

        stories, queries, answers = zip(*test_records)

        self._vocab = vocab
        self._vocab_size = vocab_size
        self._word_idx = word_idx
        self._idx_word = idx_word
        self.story_maxlen = story_maxlen
        self.query_maxlen = query_maxlen

        self._train_records = train_records
        self._test_records = test_records
        self._lookup = dict(word_idx) # deal with null cases if necessary

        self.stories = stories
        self.answers = answers

    def deindex_sentence(self, ary, prettify=True):
        """
        Take a list of ints and return a sentence of words
        :param ary: array-like, List of ints (vectorized sentence)
        :param prettify: Clean up the sentence, e.g. trim extra spaces, add line breaks
        :return: Sentence
        :rtype: str
        """
        sentence = []
        for scalar in ary:
            try:
                word = self.idx_word[scalar]
                if word:
                    sentence.append(word)
            except KeyError:
                print('Index not found in vocab: {}'.format(scalar))

        sentence = ' '.join(sentence)
        if prettify: # just tidy up a bit
            sentence = sentence.replace(' . ', '.\n').replace(' .', '.')
        return sentence

    def vectorize_all(self, datatype='train'):
        """
        Vectorize all items in the dataset
        :param datatype: {'train'|'test'} specify the dataset to use
        :return: (stories, queries, answers) each is a numpy array
        :rtype: tuple
        """
        if datatype == 'train':
            data = self.train_records
        elif datatype == 'test':
            data = self.test_records
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


    def vectorize_story(self, story):
        """
        Take a "story" and convert it to a sequence of ints using the vocab list
        :param story:
        :type story: list
        :return: list of ints
        """
        story = [self[w] for w in story]
        return pad_sequences([story], maxlen=self.story_maxlen) # note: this expects a sequence

    def vectorize_query(self, query, verbose=False):
        """
        Take a query as a sentence string and return the vector in int-list form
        :param query:
        :type query: str
        :param verbose:
        :return: list of ints
        """
        query = query.replace('?', ' ?')
        query = query.split(' ')
        exclude = ['', ' ']
        query = [q for q in query if q not in exclude]
        query = [self[q] for q in query]
        if verbose: print('<v>Vectorize_query(): {}'.format(query))
        queryvec = pad_sequences([query], maxlen=self.query_maxlen)
        return queryvec

    def devectorize_ans(self, ansvec, show_conf=False):
        """
        Take a vector from NN answer and convert it back to word form
        :param ansvec: n-dim vector, n=vocab size
        :param show_conf: print out the confidence of the top few potential matches
        :return:
        """
        idx = np.argmax(ansvec)
        if show_conf:
            conf = list(ansvec.ravel())
            vocab = [self.idx_word[i] for i in range(len(conf))]
            df = pd.DataFrame(list(zip(vocab, conf  )), columns=['vocab', 'conf'])
            df = df.sort_values(by='conf', ascending=False)
            df['conf'] = pd.Series(["{0:.2f}%".format(val * 100) for val in df['conf']], index=df.index)

            print(df.head().to_string(index=False))
        return self.idx_word[idx], ansvec.ravel()[idx]

    def format_story(self, story):
        print('-' * 30)
        print(' '.join(story).replace(' . ', '.\n').replace(' .', '.'))
        print('-' * 30)

    def get_random_story(self, show=False):
        """Migrating this over to the StoryHandler, where it belongs"""
        story = np.random.choice(self.stories)
        if show:
           self.format_story(story)
        return story

    @property
    def vocab(self): return self._vocab

    @property
    def vocab_size(self): return self._vocab_size

    @property
    def word_idx(self): return self._word_idx

    @property
    def idx_word(self): return self._idx_word

    @property
    def train_records(self): return self._train_records

    @property
    def test_records(self): return self._test_records

    @property
    def lookup(self): return self._lookup

    def __getitem__(self, item):
        """Allows us to use the vectorizer object itself to do lookups. Clever, perhaps too clever.
        Only does word_to_index lookups. index_to_word lookups must be invoked with self.idx_word
        If allow_case_insensitive is specified, try to do a match with all lower case.
        If that fails, flag the error."""
        try:
            return self.lookup[item]
        except KeyError:
            pass
        if self.allow_case_insensitive:
            correctitem = matchnocase(item, self.word_idx)
            try:
                return self.lookup[correctitem]
            except KeyError:
                pass
        if self.allow_softmatch:
            correctitem = softmatch(item, self.word_idx, lower=True, cutoff=2.)
            try:
                return self.lookup[correctitem]
            except KeyError:
                pass
        # fallthrough condition. Key not found with soft matches
        if self.ignore_keyerror:
            print('<!> Value not found in lookup: {}'.format(item))
            return 0

        else:
            raise KeyError('Value not found in lookup: {}'.format(item))

