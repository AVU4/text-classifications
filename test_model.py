import numpy as np
import sklearn.metrics as metrics
from tensorflow.keras.models import model_from_json
from TEXT_CNN.text_cnn import TextCNN
from data_holder import Corpus
from utils import get_y_values
from utils import load_data_test
from utils import load_model
from tensorflow.keras.optimizers import Adam
from BLSTM.attention import AttentionWithContext
from sklearn.metrics import accuracy_score

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 300000
EMBEDDING_DIM = 200
LEARNING_RATE = 0.0001
DATA_DIR = '/home/avu/Pycharm/text-classification-models/data/'

def buildBLSTM():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json, {'AttentionWithContext': AttentionWithContext})

    model.load_weights('model.h5')
    print("Loaded model from disk")

    model.summary()

    adam = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=[accuracy_score])

    return model


if __name__ == "__main__":

    # model = TextCNN(MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, EMBEDDING_DIM, class_num=147)
    # model = load_model(model, LEARNING_RATE)
    model = buildBLSTM()
    corpus = Corpus(DATA_DIR + 'test_set.csv', DATA_DIR + 'test_set_labels_small.csv')

    X_test, y_test = load_data_test(corpus, MAX_SEQUENCE_LENGTH)
    X_test[np.isnan(X_test)] = 0

    array_split = np.array_split(X_test, 2)
    first_part = array_split[0]
    second_part = array_split[1]
    first_part = get_y_values(first_part, model)
    second_part = get_y_values(second_part, model)
    y_test_prediction = np.concatenate((first_part, second_part))

    y_test_original = []
    for y in y_test:
        y_test_original.append(y.argmax())

    accuracy_score = metrics.accuracy_score(y_test_original, y_test_prediction)

    print('Точность = ' + str(accuracy_score))
