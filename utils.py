import json

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

from encoder import NpEncoder


def load_model(model, learning_rate):
    model.build(input_shape=(112676, 1000))
    model.load_weights('model.h5')
    print("Loaded model from disk")

    model.summary()

    adam = Adam(learning_rate=learning_rate)
    model.compile(adam, 'binary_crossentropy', metrics=['accuracy'])

    return model


def get_y_values(X_test, model):
    predictions = model.predict(X_test)

    y_test_prediction = []
    for prediction in predictions:
        max_class = prediction.argmax()
        y_test_prediction.append(max_class)
    return y_test_prediction


def handle_y_values(y_data, classes):
    y_output = []
    for y in y_data:
        array = np.zeros(len(classes))
        idx = classes[y]
        array[idx] = 1
        y_output.append(array)
    return np.array(y_output)


def load_data_test(set, max_sequence_length):
    class_file = open('class_dict.json', 'r')
    class_json = json.load(class_file)

    X_data = []
    y_data = []

    counter = 0
    for c, (vector, target) in enumerate(set):
        if target[0] in class_json:
            X_data.append(vector)
            y_data.append(target[0])
            counter += 1

    file = open('tokenizer.json', 'r')
    config = json.loads(file.read())
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(config)

    X_data = tokenizer.texts_to_sequences(X_data)

    X_data = pad_sequences(X_data,
                           maxlen=max_sequence_length,
                           padding='post',
                           truncating='post',
                           dtype='float32')

    y_data_int = handle_y_values(y_data, class_json)

    return X_data, y_data_int


def get_dataset(train_set, max_words_number, max_sequence_length):
    X_data = []
    y_data = []
    for c, (vector, target) in enumerate(train_set):
        labels = target[0].split(' ')
        if len(labels) == 1:
            X_data.append(vector)
            y_data.append(target[0])

    print((len(X_data), 'training examples'))

    labels = set()

    for y in y_data:
        labels.add(y)

    nb_classes = len(labels)
    print((nb_classes, 'classes'))
    class_dict = {value: idx for idx, value in enumerate(labels)}

    with open('class_dict.json', 'w') as fp:
        json.dump(class_dict, fp, cls=NpEncoder)
    print('Exported class dictionary')

    y_data = handle_y_values(y_data, class_dict)

    tokenizer = Tokenizer(num_words=max_words_number,
                          oov_token=1)
    tokenizer.fit_on_texts(X_data)
    X_data = tokenizer.texts_to_sequences(X_data)

    X_data = pad_sequences(X_data,
                           maxlen=max_sequence_length,
                           padding='post',
                           truncating='post',
                           dtype='float32')
    print(('Shape of data tensor:', X_data.shape))

    word_index = tokenizer.word_index
    print(('Found %s unique tokens' % len(word_index)))
    with open('word_index.json', 'w') as fp:
        json.dump(word_index, fp)
    print('Exported word dictionary')

    with open('tokenizer.json', 'w') as file:
        tokenizer_json = tokenizer.to_json()
        json.dump(tokenizer_json, file)

    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data,
                                                      train_size=0.8,
                                                      test_size=0.2,
                                                      random_state=42)

    return X_train, X_val, y_train, y_val, nb_classes, word_index
