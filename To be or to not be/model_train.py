# imports
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from pickle import dump


# get the data from corpus.txt
data = str()
with open('corpus.txt', encoding='utf-8') as corpus_file:
    data = corpus_file.read()

tokenizer = Tokenizer()


# pre-preprocess the loaded data
def data_preprocessing(data):
    corpus = data.lower().split("\n")
    tokenizer.fit_on_texts(corpus)
    sequences = tokenizer.texts_to_sequences(corpus)
    vocab_len = len(tokenizer.word_index) + 1
    max_sequence_len = max([len(x) for x in sequences])
    sequences = np.array(pad_sequences(sequences, maxlen=max_sequence_len,
                                       padding='pre'))
    x_train, y_train = sequences[:, :-1], sequences[:, -1]

    return (x_train, y_train, max_sequence_len, vocab_len)


# model
def model(x_train, y_train, max_sequence_len, vocab_len):
    input_len = max_sequence_len - 1

    model = Sequential()
    model.add(Embedding(vocab_len, 40, input_length=input_len))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_len, activation='softmax'))

    model.summary()

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=6, batch_size=128, verbose=1)

    return model


x_train, y_train, max_len, vocab_len = data_preprocessing(data)
model = model(x_train, y_train, max_len, vocab_len)

# save model and tokenizer details after training
model.save('model.h5')
dump(tokenizer, open('tokenizer.pkl', 'wb'))
