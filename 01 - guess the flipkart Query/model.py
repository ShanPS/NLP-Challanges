# imports
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
import numpy as np

# set seed for consistency with results
np.random.seed(6699)

# define hyper-parameters
num_classes = 20
batch_size = 1
epochs = 8

# lists for taking texts and corresponding class
texts = []
labels = []

# 20 classes
classes = ['axe deo', 'best-seller books', 'calvin klein', 'camcorder',
           'camera', 'chemistry', 'chromebook', 'c programming',
           'data structures algorithms', 'dell laptops', 'dslr canon',
           'mathematics', 'nike-deodrant', 'physics', 'sony cybershot',
           'spoken english', 'timex watch', 'titan watch', 'tommy watch',
           'written english']

# vocabulary used for BOW representation of documents
vocab = ['1', '2', '3', '4', '5', '6', '135mm', '1dx', '4gb', '500gb', '55mm',
         '600d', '60d', '700d', '7d', 'academic', 'advanced', 'aieee',
         'algorithms', 'apollo', 'applications', 'approach', 'aqua',
         'available', 'axe', 'backpack', 'bag', 'bank', 'banking', 'banks',
         'biology', 'body', 'bogoli', 'book', 'books', 'botany', 'boxed',
         'boxset', 'c', 'calvin', 'camcorder', 'camera', 'canon', 'case',
         'cb001', 'cbse', 'champs', 'chapterwise', 'chemical', 'chemistry',
         'chromebook', 'ci3', 'ck', 'class', 'clerk', 'collection', 'combo',
         'combos', 'common', 'complete', 'control', 'coolpix', 'course',
         'cover', 'cwe', 'cx220e', 'cyber', 'cybershot', 'data', 'dcb', 'de',
         'definitions', 'dell', 'deodorant', 'depth', 'digiflip', 'dsc',
         'eau', 'edition', 'ef', 'english', 'eos', 'exam', 'examination',
         'gen', 'go', 'google', 'grammar', 'guide', 'h200', 'h300', 'h70',
         'hc', 'hdc', 'hdr', 'hilfiger', 'ibps', 'ice', 'in2u', 'inspiron',
         'institute', 'intermediate', 'java', 'karishma', 'kit', 'klein',
         'l27', 'l3', 'laptop', 'laptops', 'lens', 'logic', 'longman',
         'management', 'mariner', 'mathematics', 'mechanics', 'mind', 'ml',
         'mm', 'modern', 'n150', 'never', 'nike', 'nikon', 'object', 'octane',
         'officers', 'oriented', 'panasonic', 'personnel', 'phut', 'physics',
         'pj230e', 'po', 'point', 'power', 'practical', 'pricey',
         'probationary', 'programming', 'reports', 'rrb', 'rural', 's15',
         's18', 'sdr', 'secret', 'selection', 'selections', 'set', 'shoot',
         'shot', 'slr', 'sony', 'spoken', 'spray', 'statistical', 'stories',
         'structures', 'subconscious', 'success', 'thermodynamics', 'timex',
         'titan', 'tm900', 'toilette', 'tommy', 'trainees', 'ubuntu', 'v110',
         'vostro', 'w610', 'w710', 'watch', 'workbook', 'writing', 'written',
         'wx300', 'wx50', 'zoology']

# dicts for mapping from index to class and vice-versa
index_to_class = dict(enumerate(classes))
class_to_index = {k: v for v, k in index_to_class.items()}

# get the training data and labels
with open('training.txt') as train_file:
    next(train_file)    # skips header
    for row in train_file:
        for c in classes:
            if(row.endswith(c + '\n')):
                texts.append(row[:-(len(c)+1)])
                labels.append(class_to_index[c])
                break

# vectorizer for vectorizing documents as BOW representation
vectorizer = CountVectorizer(token_pattern=r'\w{1,}')
vectorizer.fit(vocab)

# verctorize train data
x_train = vectorizer.transform(texts).toarray()
# one hot - train labels
y_train = to_categorical(labels, num_classes)

# for validation let's use given sample-input and sample-output data
x_test = []
y_test = []
with open('sample-input.txt') as test_texts_file:
    next(test_texts_file)
    for row in test_texts_file:
        x_test.append(row[:-1])
with open('sample-output.txt') as test_labels_file:
    for row in test_labels_file:
        y_test.append(class_to_index[row[:-1]])

# verctorize test data
x_test = vectorizer.transform(x_test).toarray()
# one hot - test labels
y_test = to_categorical(y_test, num_classes)


# model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(len(vocab),)))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(x_test, y_test))

# evaluate using gieven sample data and labels
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# for prediction use 'model.predict(data)', then take max-valued
# index and map it corresponding class using 'index_to_class'

# prediction on sample data for demonstration
predictions = model.predict(x_test.reshape(-1, len(vocab)))
predictions = np.argmax(predictions, axis=1)
predictions_class = []
for i in range(len(predictions)):
    predictions_class.append(index_to_class[predictions[i]])
    print(predictions_class[i])
