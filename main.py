# python libs
import argparse

# local libs
import menu
import models
import preprocess


def set_arg_parser():
    parser = argparse.ArgumentParser(description='Process eeg data. See docs/main.txt for more info')
    parser.add_argument("-v", "--verbose", action="store_true",
                    help="output verbosity")

    return parser

def query_model(query=None, model=None, vectorizer=None):
    queryvec = vectorizer


class StoryHandler:
    def __init__(self, dmn, vectorizer):
        self.dmn = dmn
        self.vectorizer = vectorizer

    def get_random_story(self):
        story = self.vectorizer.get_random_story()
        self.story = story
        return story


if __name__ == '__main__':
    parser = set_arg_parser()
    args = parser.parse_args()
    verbose = args.verbose
    if verbose: print('<v> Verbose print on')

    modelfile = 'dmn00.hdf5'
    ve = preprocess.BabiVectorizer()
    dmn = models.DeepMemNet(vocab_size=ve.vocab_size, story_maxlen=ve.story_maxlen, query_maxlen=ve.query_maxlen)
    dmn.model.load_weights(modelfile)
    if verbose: print('<v> Loaded model: {}'.format(modelfile))

    handler = StoryHandler(dmn, ve)

    mainitems = [['1', 'Load Random Story', handler.get_random_story],
                 ['2', 'Query', lambda: 2]]
    mainmenu = menu.Menu('00', '', mainitems)

    handler.get_random_story()
    while True:
        story = handler.story
        ve.format_story(story) # Display the current story
        reply = mainmenu()
        if verbose: print('<d>Menu returned: |{}| {}'.format(reply, type(reply)))
        if reply == 'q':
            break
        if reply == 2:
            print('Enter a query: ')
            query = input('Enter a query: ')
            queryvec = ve.vectorize_query(query)
            storyvec = ve.vectorize_story(story)
            ans = dmn.query(storyvec, queryvec)
            ans_word = ve.devectorize_ans(ans)
            print('Predicted answer:\n>> {}'.format(ans_word))
            print('_'*30)

