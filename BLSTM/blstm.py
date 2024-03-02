import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from BLSTM.attention import AttentionWithContext

EMBEDDING_FILE = '/home/avu/Pycharm/Document-Classifier-LSTM/glove.6B.200d.txt'


def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, "float32")
    y_pred = tf.cast(tf.round(y_pred), "float32")  # implicit 0.5 threshold via tf.round
    y_correct = y_true * y_pred

    sum_true = tf.reduce_sum(y_true, axis=1)
    sum_pred = tf.reduce_sum(y_pred, axis=1)
    sum_correct = tf.reduce_sum(y_correct, axis=1)

    precision = sum_correct / sum_pred
    recall = sum_correct / sum_true
    f_score = 2 * precision * recall / (precision + recall)
    f_score = tf.where(tf.math.is_nan(f_score), tf.zeros_like(f_score), f_score)

    return tf.reduce_mean(f_score)


def prepare_embeddings(wrd2id, max_words_number, EMBEDDING_DIM):
    vocab_size = max_words_number
    print(("Found %s words in the vocabulary." % vocab_size))

    embedding_idx = {}
    glove_f = open(EMBEDDING_FILE)
    for line in glove_f:
        values = line.split()
        wrd = values[0]
        coefs = np.asarray(values[1:],
                           dtype='float32')
        embedding_idx[wrd] = coefs
    glove_f.close()
    print(("Found %s word vectors." % len(embedding_idx)))

    embedding_mat = np.random.rand(vocab_size + 1, EMBEDDING_DIM)

    wrds_with_embeddings = 0
    # Keep the MAX_NB_WORDS most frequent tokens.
    for wrd, i in wrd2id.items():
        if i > vocab_size:
            continue

        embedding_vec = embedding_idx.get(wrd)
        # words without embeddings will be left with random values.
        if embedding_vec is not None:
            wrds_with_embeddings += 1
            embedding_mat[i] = embedding_vec

    print((embedding_mat.shape))
    print(('Words with embeddings:', wrds_with_embeddings))

    return embedding_mat, vocab_size


def build_model(nb_classes,
                word_index,
                max_words_num,
                embedding_dim,
                seq_length,
                stamp):
    embedding_matrix, nb_words = prepare_embeddings(word_index, max_words_num, embedding_dim)

    input_layer = Input(shape=(seq_length,),
                        dtype='int32')

    embedding_layer = Embedding(input_dim=nb_words + 1,
                                output_dim=embedding_dim,
                                input_length=seq_length,
                                weights=[embedding_matrix],
                                embeddings_regularizer=regularizers.l2(0.00),
                                trainable=True)(input_layer)

    drop1 = SpatialDropout1D(0.3)(embedding_layer)

    lstm_1 = Bidirectional(LSTM(128, name='blstm_1',
                                activation='tanh',
                                recurrent_activation='sigmoid',
                                recurrent_dropout=0.0,
                                dropout=0.5,
                                kernel_initializer='glorot_uniform',
                                return_sequences=True),
                           merge_mode='concat')(drop1)
    lstm_1 = BatchNormalization()(lstm_1)

    att_layer = AttentionWithContext()(lstm_1)

    drop3 = Dropout(0.5)(att_layer)

    predictions = Dense(nb_classes, activation='sigmoid')(drop3)

    model = Model(inputs=input_layer, outputs=predictions)

    adam = Adam(learning_rate=0.001)

    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=[accuracy_score])

    model.summary()
    print(stamp)

    return model
