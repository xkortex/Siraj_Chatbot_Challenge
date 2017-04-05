from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate
from keras.layers import LSTM, TimeDistributed
from keras.layers import Conv1D
from keras.layers.wrappers import Bidirectional

class ConfigurableNetwork:
    defaultfile = 'default.cfg'
    def __init__(self, modelname):
        self._modelname = modelname

        # Set up the environment. load config file if it is there, if not, then create it form default
        # Load the weights

        # existential question: should this contain all of the handling for fitting and callbacks?
        # yeah I think that makes sense. One stop shop for configs, weights, and logging.


    def setup(self):
        pass

    @property
    def modelname(self): return self._modelname

class DeepMemNet:
    """
    DeepMemNet for the Facebook bAbI context task.

    Model notes:
    Single context task:
      Regular LSTM (accuracy/val_acc):
        Run 1
            47/50% @ 7 epochs
            72/70% @ 49 epochs
            86/80% @ 61 epochs
            95/90% @ 88 epochs
        Run 2



      Bidirectional LSTM:
        Run 1
            50%/50% @ 6
            81%/80% @ 34
            87%/86% @ 48 (peak valacc)
            90%/86% @ 60 epochs - minor overfitting

        I think parameters were not configured right, these might be Single LSTM! need to rerun everything >_<
        Run 2
            73/71% @ 44
            83/80% @ 53
            94/90% @ 75
        Run 3
            71/70% @ 41
            83/80% @ 50
            94/90% @ 76

      Bidirectional + extra forward LSTM:
        55%/??% @ 80 (stalls)

      TDDense + Bidirectional:
        Run 1:
            76/73% @ 37
            80/80% @ 41
            91/90% @ 54 - new record!!

    Double Context task:
      Regular:
        50%/??% @ 35 epochs
        67%/??% @ 80 epochs
        70%/??% @ 100
        80%/??% @ 192
        84.7%/??% @ 260

      Bidirectional:
        50%/??% @ 26 epochs
        70%/??% @ 48 epochs - improvement!
        80%/??% @ 68 epochs - super improvement!
        90%/??% @ 110 epochs - smokin'!
        95%/??% @ 148 epochs - starting to level off
        97%/??% @ 200 epochs - i think it's starting to overfit
    """
    # todo: add performance logging
    def __init__(self, vocab_size=22, story_maxlen=68, query_maxlen=4, n_lstm=32, bidirect=True, tdd=True,
                 matchconv=False, permute=False):
        """
        DeepMemNet

        Param note - changing parameters will require new model file (duh) - this isn't automatic yet
        :param vocab_size:
        :param story_maxlen:
        :param query_maxlen:
        :param n_lstm:
        :param bidirect:
        """

        # todo: config file for model hyperparams with logging link

        self.vocab_size = vocab_size
        self.story_maxlen = story_maxlen
        self.query_maxlen = query_maxlen
        # placeholders
        input_sequence = Input((story_maxlen,), name='InputSeq')
        question = Input((query_maxlen,), name='Question')

        # Encoders - initial encoders are pretty much just projecting the input into a useful space
        # not much need to optimize here really
        input_encoder_m = Sequential(name='InputEncoderM')
        input_encoder_m.add(Embedding(input_dim=vocab_size,
                                      output_dim=64, name='InEncM_Embed'))
        input_encoder_m.add(Dropout(0.3))
        # output: (samples, story_maxlen, embedding_dim)

        # embed the input into a sequence of vectors of size query_maxlen
        input_encoder_c = Sequential(name='InputEncoderC')
        input_encoder_c.add(Embedding(input_dim=vocab_size,
                                      output_dim=query_maxlen, name='InEncC_Embed'))
        input_encoder_c.add(Dropout(0.3))
        # output: (samples, story_maxlen, query_maxlen)

        # embed the question into a sequence of vectors
        question_encoder = Sequential(name='QuestionEncoder')
        question_encoder.add(Embedding(input_dim=vocab_size,
                                       output_dim=64,
                                       input_length=query_maxlen, name='QuesEnc_Embed'))
        question_encoder.add(Dropout(0.3))
        # output: (samples, query_maxlen, embedding_dim)

        # encode input sequence and questions (which are indices)
        # to sequences of dense vectors
        input_encoded_m = input_encoder_m(input_sequence)
        input_encoded_c = input_encoder_c(input_sequence)
        question_encoded = question_encoder(question)

        # compute a 'match' between the first input vector sequence
        # and the question vector sequence
        # shape: `(samples, story_maxlen, query_maxlen)`
        match = dot([input_encoded_m, question_encoded], axes=(2, 2), name='Match')
        match = Activation('softmax')(match)

        if matchconv:
            match = Conv1D(query_maxlen, 4, padding='same')(match)

        # add the match matrix with the second input vector sequence
        response = add([match, input_encoded_c], name='ResponseAdd')  # (samples, story_maxlen, query_maxlen)
        response = Permute((2, 1), name='ResponsePermute')(response)  # (samples, query_maxlen, story_maxlen)

        # concatenate the match matrix with the question vector sequence
        answer = concatenate([response, question_encoded], name='AnswerConcat')

        # Trying to feed in the long axis as the timestep causes the GPU to get very angry.
        # It would appear it causes it to start thrashing memory
        if permute:
            answer = Permute((2, 1), name='AnswerPermute')(answer)  # (samples, story_maxlen, query_maxlen)


        # Let's try with a time distributed dense before the RNN
        if tdd:
            answer = TimeDistributed(Dense(n_lstm, name='Answer_TDD'))(answer)

        # Bidirectional LSTM for better context recognition, plus an additional one for flavor
        lstm_rev = Bidirectional(LSTM(n_lstm, return_sequences=True, name='Ans_LSTM_reverse'))
        lstm_for = Bidirectional(LSTM(n_lstm, return_sequences=False, name='Ans_LSTM_forward'))
        if bidirect:
            answer = lstm_rev(answer)  # "reverse" pass goes first
        answer = lstm_for(answer)
        # answer = LSTM(n_lstm, name='Ans_LSTM_3)(answer) # Extra LSTM completely runs out of steam at 55% acc! Bidirectional seems to help



        # one regularization layer -- more would probably be needed.
        answer = Dropout(0.3, name='Answer_Drop')(answer)
        answer = Dense(vocab_size, name='Answer_Dense')(answer)  # (samples, vocab_size)
        # we output a probability distribution over the vocabulary
        answer = Activation('softmax')(answer)

        # build the final model
        model = Model([input_sequence, question], answer)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model

    def query(self, storyvec, queryvec):
        storyvec = storyvec.reshape((-1, self.story_maxlen))
        queryvec = queryvec.reshape((-1, self.query_maxlen))
        ans = self.model.predict([storyvec, queryvec])
        return ans