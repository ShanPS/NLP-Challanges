# imports
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from pickle import load


# load the model and tokenizer
model = load_model('model.h5')
tokenizer = load(open('tokenizer.pkl', 'rb'))

# target words
targets = ['am', 'are', 'were', 'was', 'is', 'been', 'being', 'be']
# corresponding indices as per tokenizer.word_index
target_indices = []
for i in targets:
    target_indices.append(tokenizer.word_index[i])
# dictionary to map target indices to target words
target_dict = dict(zip(target_indices, targets))


# read the data to do the predictions
with open('sample-input.txt') as data_file:
    count = int(next(data_file))
    data = data_file.read()

# pre-process the data
data = data.split('----')
data = data[:-1]
max_sequence_len = model.layers[0].input_shape[1]
texts = []
for text in data:
        token_list = tokenizer.texts_to_sequences([text])[0]
        texts.append(token_list)
texts = np.array(pad_sequences(texts, maxlen=max_sequence_len, padding='pre'))

# predict the targets
predictions = model.predict(texts)

# print the results
results = []
for predicted in predictions:
    max_prob = 0
    for index in target_indices:
        if(predicted[index-1] > max_prob):
            max_prob = predicted[index-1]
            result = target_dict[index]
    results.append(result)

for result in results:
    print(result)
