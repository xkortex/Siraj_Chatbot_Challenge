from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate
from keras.layers import LSTM
from keras.layers.wrappers import Bidirectional

class DeepMemNet:
    """
    DeepMemNet for the Facebook bAbI context task.

    Model notes:
    Single context task:
      Num Epochs to reach 50% acc:
        Regular LSTM: 10 epochs
      Num Epochs to reach 90% acc:
        Regular LSTM: 85 epochs
        Bidirectional 60 epochs - improvement!
        Bidirectional + Extra forward LSTM: stalls at 55%

    Double Context task:
      Num Epochs to reach 50% acc:
        Regular LSTM: 35 epochs
    """
    def __init__(self, vocab_size=22, story_maxlen=68, query_maxlen=4, n_lstm=32,):

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

        # add the match matrix with the second input vector sequence
        response = add([match, input_encoded_c], name='ResponseAdd')  # (samples, story_maxlen, query_maxlen)
        response = Permute((2, 1), name='ResponsePermute')(response)  # (samples, query_maxlen, story_maxlen)

        # concatenate the match matrix with the question vector sequence
        answer = concatenate([response, question_encoded], name='AnswerConcat')

        # Bidirectional LSTM for better context recognition, plus an additional one for flavor
        # lstm_1 = Bidirectional(LSTM(n_lstm, return_sequences=True, name='Ans_LSTM_1'))
        lstm_2 = Bidirectional(LSTM(n_lstm, return_sequences=False, name='Ans_LSTM_2'))
        # answer = lstm_1(answer)  # (samples, 32)
        answer = lstm_2(answer)
        # answer = LSTM(n_lstm, name='Ans_LSTM_3)(answer) # Extra LSTM completely runs out of steam at 55% acc! Bidirectional seems to help

        #
        # Interestingly

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