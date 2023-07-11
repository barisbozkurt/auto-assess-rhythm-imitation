from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow.keras.backend as K
import sys

class Siamese():
    def __init__(self,  encoder, input_shape=None, distance_metric='reg_dist'):
        assert distance_metric in ('reg_diff', 'reg_concat', 'uniform_euclidean', 'weighted_euclidean',
                                   'uniform_l1', 'weighted_l1',
                                   'dot_product', 'cosine_distance', 'concat')

        self.encoder = encoder
        self.input_shape = input_shape
        self.distance_metric = distance_metric

        self.siamese = self.build_siamese_net()

    def build_siamese_net(self):
        input_1 = layers.Input(shape=self.input_shape)
        input_2 = layers.Input(shape=self.input_shape)

        encoded_1 = self.encoder(input_1)
        encoded_2 = self.encoder(input_2)

        if self.distance_metric == 'reg_diff':
            embedded_distance = layers.Subtract(name='subtract_embeddings')([encoded_1, encoded_2])
            embedded_distance = layers.Lambda(lambda x: K.square(x))(embedded_distance)
            output = layers.Dense(8, activation='relu')(embedded_distance)
            output = layers.Dense(1, activation='relu')(output)
        elif self.distance_metric == 'reg_concat':
            concat_embedding = layers.Concatenate(name='concat_embeddings')([encoded_1, encoded_2])
            output = layers.Dense(8, activation='relu')(concat_embedding)
            output = layers.Dense(1, activation='relu')(output)
        elif self.distance_metric == 'weighted_l1':
            # This is the distance metric used in the original one-shot paper
            # https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
            embedded_distance = layers.Subtract(name='subtract_embeddings')([encoded_1, encoded_2])
            embedded_distance = layers.Lambda(lambda x: K.abs(x))(embedded_distance)
            output = layers.Dense(1, activation='sigmoid')(embedded_distance)
        elif self.distance_metric == 'uniform_euclidean':
            # Simpler, no bells-and-whistles euclidean distance
            # Still apply a sigmoid activation on the euclidean distance however
            embedded_distance = layers.Subtract(name='subtract_embeddings')([encoded_1, encoded_2])
            # Sqrt of sum of squares
            embedded_distance = layers.Lambda(
                lambda x: K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True)), name='euclidean_distance'
            )(embedded_distance)
            output = layers.Dense(1, activation='sigmoid')(embedded_distance)
        elif self.distance_metric == 'concat':
            concat_embedding = layers.Concatenate()([encoded_1, encoded_2])
            output = layers.Dense(1, activation='sigmoid')(concat_embedding)
        elif self.distance_metric == 'cosine_distance':
            raise NotImplementedError
            # cosine_proximity = layers.Dot(axes=-1, normalize=True)([encoded_1, encoded_2])
            # ones = layers.Input(tensor=K.ones_like(cosine_proximity))
            # cosine_distance = layers.Subtract()([ones, cosine_proximity])
            # output = layers.Dense(1, activation='sigmoid')(cosine_distance)
        else:
            raise NotImplementedError

        siamese = Model(inputs=[input_1, input_2], outputs=output)

        return siamese


class Models():
    def __init__(self, model_name='CNN2D', embedding_dimension=24, filters=12, units=24, input_shape=None, dropout=0.05):
        assert model_name in ('LSTM', 'CNN1D', 'CNN1D_RNN', 'CNN2D_RNN', 'CNN1D_RNN_pretrained','CNN2D', 'FF', 'LSTM_seq2seq', 'CNN1D_RNN_seq2seq', 'CNN2D_RNN_seq2seq')

        self.model_name = model_name
        self.embedding_dimension = embedding_dimension
        self.filters = filters
        self.units = units
        self.input_shape = input_shape
        self.dropout = dropout

        if model_name == 'LSTM':
            self.encoder = self.get_LSTM_encoder()
        if model_name == 'CNN1D':
            self.encoder = self.get_CNN1D_encoder()
        if model_name == 'CNN1D_RNN':
            self.encoder = self.get_CNN1D_RNN_encoder()
        if model_name == 'CNN2D_RNN':
            self.encoder = self.get_CNN2D_RNN_encoder()
        if model_name == 'CNN1D_RNN_pretrained':
            self.encoder = self.get_CNN1D_RNN_pretrained_encoder()
        if model_name == 'CNN2D':
            self.encoder = self.get_CNN2D_encoder()
        if model_name == 'FF':
            self.encoder = self.get_melody_FF_encoder()
        if model_name == 'LSTM_seq2seq':
            self.encoder = self.get_LSTM_seq2seq()
        if model_name == 'CNN1D_RNN_seq2seq':
            self.encoder = self.get_CNN1D_RNN_seq2seq()
        if model_name == 'CNN2D_RNN_seq2seq':
            self.encoder = self.get_CNN2D_RNN_seq2seq()

    def get_CNN1D_RNN_pretrained_encoder(self):
        epoch = 200
        model_save_path = '../../data/models/rhthym_onset_dnn'

        encoder = keras.models.load_model('{}/onset_model_epoch-{}.hdf5'.format(model_save_path, epoch))

        encoder.add(layers.Flatten())
        encoder.add(layers.Dense(self.units, activation='relu'))
        #encoder.add(layers.Dense(self.units * 2, activation='relu'))
        encoder.add(layers.Dense(self.embedding_dimension, activation='relu'))

        encoder.layers[0].trainable = False
        encoder.layers[1].trainable = False
        encoder.layers[2].trainable = False
        encoder.layers[3].trainable = False
        encoder.layers[4].trainable = False
        encoder.layers[5].trainable = False

        return encoder

    def get_CNN1D_RNN_encoder(self):
        encoder = Sequential()

        # Initial conv
        if self.input_shape is None:
            encoder.add(layers.Conv1D(filters=self.filters, kernel_size=7, padding='same', activation='relu'))
        else:
            encoder.add(layers.Conv1D(filters=self.filters, kernel_size=7, padding='same', activation='relu',
                                      input_shape=self.input_shape))
        encoder.add(layers.BatchNormalization())

        # Further convs
        encoder.add(layers.Conv1D(filters=self.filters * 2, kernel_size=5, padding='same', activation='relu'))
        encoder.add(layers.BatchNormalization())

        encoder.add(
            layers.LSTM(units=self.units, return_sequences=False, activation='relu'))

        #encoder.add(layers.TimeDistributed(layers.Dense(1, activation='relu')))
        #encoder.add(layers.Flatten())
        encoder.add(layers.Dense(self.embedding_dimension, activation='relu'))

        return encoder

    def get_melody_FF_encoder(self):
        encoder = Sequential()
        if self.input_shape is None:
            # In this case we are using the encoder as part of a siamese network and the input shape will be determined
            # automatically based on the input shape of the siamese network
            encoder.add(layers.Dense(self.units, activation='relu'))
        else:
            # In this case we are using the encoder to build a classifier network and the input shape must be defined
            encoder.add(layers.Dense(self.units, input_shape=self.input_shape, activation='relu'))

        encoder.add(layers.Dense(self.units/2, activation='relu'))
        encoder.add(layers.Dense(self.embedding_dimension, activation='relu'))

        return encoder

    def get_FF_encoder(self):
        encoder = Sequential()
        if self.input_shape is None:
            # In this case we are using the encoder as part of a siamese network and the input shape will be determined
            # automatically based on the input shape of the siamese network
            encoder.add(layers.Dense(self.units, activation='relu'))
        else:
            # In this case we are using the encoder to build a classifier network and the input shape must be defined
            encoder.add(layers.Dense(self.units, activation='relu', input_shape=self.input_shape))

        encoder.add(layers.Dense(self.units*2, activation='relu'))
        encoder.add(layers.Dense(self.embedding_dimension, activation='relu'))

        return encoder

    def get_CNN1D_RNN_seq2seq(self):
        encoder = Sequential()

        # Initial conv
        if self.input_shape is None:
            encoder.add(layers.Conv1D(filters=self.filters, kernel_size=9, padding='same', activation='relu'))
        else:
            encoder.add(layers.Conv1D(filters=self.filters, kernel_size=9, padding='same', activation='relu',
                                      input_shape=self.input_shape))
        encoder.add(layers.BatchNormalization())

        # Further convs
        encoder.add(layers.Conv1D(filters=self.filters * 2, kernel_size=7, padding='same', activation='relu'))
        encoder.add(layers.BatchNormalization())

        encoder.add(
            layers.LSTM(units=self.units, return_sequences=True, activation='relu'))

        #encoder.add(
        #    layers.TimeDistributed(layers.Dense(1, activation='sigmoid')))
        encoder.add(layers.TimeDistributed(layers.Dense(1, activation='relu')))

        return encoder

    def get_CNN2D_RNN_seq2seq(self):
        encoder = Sequential()
        # Initial conv
        if self.input_shape is None:
            encoder.add(layers.Conv2D(filters=self.filters, kernel_size=5, padding='same', activation='relu'))
        else:
            encoder.add(layers.Conv2D(filters=self.filters, kernel_size=5, padding='same', activation='relu',
                                      input_shape=self.input_shape))
        encoder.add(layers.BatchNormalization())

        encoder.add(layers.Conv2D(filters=self.filters, kernel_size=3, padding='same', activation='relu'))
        encoder.add(layers.BatchNormalization())

        encoder.add(
            layers.LSTM(units=self.units, return_sequences=True, activation='relu'))

        encoder.add(
            layers.TimeDistributed(layers.Dense(1, activation='sigmoid')))

        return encoder

    def get_LSTM_seq2seq(self):
        encoder = Sequential()
        if self.input_shape is None:
            # In this case we are using the encoder as part of a siamese network and the input shape will be determined
            # automatically based on the input shape of the siamese network
            encoder.add(
                layers.LSTM(units=self.units, return_sequences=True, activation='relu'))
        else:
            # In this case we are using the encoder to build a classifier network and the input shape must be defined
            encoder.add(
                layers.LSTM(units=self.units, input_shape=self.input_shape, return_sequences=True, activation='relu'))

        encoder.add(
            layers.LSTM(units=self.units, return_sequences=True, activation='relu'))
        #encoder.add(
        #    layers.LSTM(units=self.units, return_sequences=True, activation='relu'))

        encoder.add(
            layers.TimeDistributed(layers.Dense(1, activation='sigmoid')))

        return encoder

    def get_LSTM_encoder(self):
        encoder = Sequential()
        if self.input_shape is None:
            # In this case we are using the encoder as part of a siamese network and the input shape will be determined
            # automatically based on the input shape of the siamese network
            encoder.add(
                layers.LSTM(units=self.units, dropout=self.dropout, return_sequences=True, activation='relu'))
        else:
            # In this case we are using the encoder to build a classifier network and the input shape must be defined
            encoder.add(
                layers.LSTM(units=self.units, dropout=self.dropout, input_shape=self.input_shape, return_sequences=True, activation='relu'))

        encoder.add(
            layers.LSTM(units=self.units, return_sequences=False, dropout=self.dropout))
        #encoder.add(layers.LSTM(units=units, return_sequences=False, dropout=dropout))
        encoder.add(
            layers.Dense(self.embedding_dimension))

        return encoder

    def get_CNN2D_classifier(self):
        encoder = Sequential()

        # Initial conv
        if self.input_shape is None:
            # In this case we are using the encoder as part of a siamese network and the input shape will be determined
            # automatically based on the input shape of the siamese network
            encoder.add(layers.Conv2D(filters=self.filters, kernel_size=5, padding='same', activation='relu'))
        else:
            # In this case we are using the encoder to build a classifier network and the input shape must be defined
            encoder.add(layers.Conv2D(filters=self.filters * 2, kernel_size=5, padding='same', activation='relu',
                                      input_shape=self.input_shape))
        encoder.add(layers.BatchNormalization())
        encoder.add(layers.SpatialDropout2D(self.dropout))
        encoder.add(layers.MaxPooling2D())

        encoder.add(layers.Conv2D(filters=self.filters, kernel_size=3, padding='same', activation='relu'))
        encoder.add(layers.BatchNormalization())
        encoder.add(layers.SpatialDropout2D(self.dropout))
        encoder.add(layers.MaxPooling2D())

        encoder.add(layers.Conv2D(filters=self.filters * 4, kernel_size=3, padding='same', activation='relu'))
        encoder.add(layers.BatchNormalization())
        encoder.add(layers.SpatialDropout2D(self.dropout))
        encoder.add(layers.MaxPooling2D())

        encoder.add(layers.Flatten())

        encoder.add(layers.Dense(self.embedding_dimension), activation='relu')
        encoder.add(layers.Dense(1, activation='relu'))

        return encoder

    def get_CNN2D_encoder(self):

        encoder = Sequential()

        # Initial conv
        if self.input_shape is None:
            # In this case we are using the encoder as part of a siamese network and the input shape will be determined
            # automatically based on the input shape of the siamese network
            encoder.add(layers.Conv2D(filters=self.filters, kernel_size=5, padding='same', activation='relu'))
        else:
            # In this case we are using the encoder to build a classifier network and the input shape must be defined
            encoder.add(layers.Conv2D(filters=self.filters*2, kernel_size=5, padding='same', activation='relu', input_shape=self.input_shape))
        encoder.add(layers.BatchNormalization())
        encoder.add(layers.SpatialDropout2D(self.dropout))
        encoder.add(layers.MaxPooling2D())

        encoder.add(layers.Conv2D(filters=self.filters, kernel_size=3, padding='same', activation='relu'))
        encoder.add(layers.BatchNormalization())
        encoder.add(layers.SpatialDropout2D(self.dropout))
        encoder.add(layers.MaxPooling2D())

        encoder.add(layers.Conv2D(filters=self.filters*4, kernel_size=3, padding='same', activation='relu'))
        encoder.add(layers.BatchNormalization())
        encoder.add(layers.SpatialDropout2D(self.dropout))
        encoder.add(layers.MaxPooling2D())

        encoder.add(layers.Flatten())

        encoder.add(layers.Dense(self.embedding_dimension))

        return encoder

    def get_CNN1D_encoder(self):
        encoder = Sequential()
        kernel_size = 5

        # Initial conv
        if self.input_shape is None:
            # In this case we are using the encoder as part of a siamese network and the input shape will be determined
            # automatically based on the input shape of the siamese network
            encoder.add(layers.Conv1D(filters=self.filters, kernel_size=kernel_size, padding='same', activation='relu'))
        else:
            # In this case we are using the encoder to build a classifier network and the input shape must be defined
            encoder.add(layers.Conv1D(filters=self.filters, kernel_size=kernel_size, padding='same', activation='relu', input_shape=self.input_shape))
        encoder.add(layers.BatchNormalization())
        encoder.add(layers.SpatialDropout1D(self.dropout))
        encoder.add(layers.MaxPool1D(4, 4))

        # Further convs
        encoder.add(layers.Conv1D(filters=self.filters*2, kernel_size=kernel_size-2, padding='same', activation='relu'))
        encoder.add(layers.BatchNormalization())
        encoder.add(layers.SpatialDropout1D(self.dropout))
        encoder.add(layers.MaxPool1D())

        encoder.add(layers.Conv1D(filters=self.filters*2, kernel_size=kernel_size-2, padding='same', activation='relu'))
        encoder.add(layers.BatchNormalization())
        encoder.add(layers.SpatialDropout1D(self.dropout))
        encoder.add(layers.MaxPool1D())

        #encoder.add(layers.Conv1D(filters=2*filters, kernel_size=5, padding='same', activation='relu'))
        #encoder.add(layers.BatchNormalization())
        #encoder.add(layers.SpatialDropout1D(dropout))
        #encoder.add(layers.MaxPool1D())

        encoder.add(layers.GlobalMaxPool1D())

        encoder.add(layers.Dense(self.embedding_dimension))

        return encoder


