from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import BLSTM.blstm
from data_holder import Corpus
from utils import get_dataset

LEARNING_RATE = 0.0001
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 300000
EMBEDDING_DIM = 200
DATA_DIR = '/home/avu/Pycharm/text-classification-models/data/'

if __name__ == "__main__":

    corpus = Corpus(DATA_DIR + 'test_set.csv', DATA_DIR + 'test_set_labels_small.csv')
    X_train, X_val, y_train, y_val, nb_classes, word_index = get_dataset(corpus, max_words_number=MAX_NB_WORDS, max_sequence_length=MAX_SEQUENCE_LENGTH)

    # model = TextCNN(MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, EMBEDDING_DIM, class_num=nb_classes)
    model = BLSTM.blstm.build_model(nb_classes, word_index, MAX_NB_WORDS, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, 'BLSTM')

    adam = Adam(learning_rate=LEARNING_RATE)
    model.compile(adam, 'binary_crossentropy', metrics=['accuracy'])

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    early_stopping = EarlyStopping(monitor='val_accuracy',
                                   patience=5,
                                   start_from_epoch=20,
                                   verbose=1,
                                   mode='max')

    model_checkpoint = ModelCheckpoint("model.h5",
                                       monitor='val_accuracy',
                                       verbose=1,
                                       save_best_only=True,
                                       mode='max',
                                       save_weights_only=True)

    model.fit(X_train, y_train,
              batch_size=128,
              epochs=100000,
              shuffle=True,
              callbacks=[early_stopping, model_checkpoint],
              validation_data=(X_val, y_val))
