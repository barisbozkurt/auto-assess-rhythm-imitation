import sys
sys.path.append('../')

from tensorflow import keras
from configuration import get_config
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers

import tensorflow.keras.layers
import tensorflow.keras.models
import tensorflow.keras.optimizers
import tensorflow.keras.datasets
import tensorflow.keras.utils
import tensorflow.keras.backend as K
from tensorflow.keras import preprocessing

config = get_config()
epoch = 200

class ScaleLayer(tensorflow.keras.layers.Layer):
    def __init__(self):
      super(ScaleLayer, self).__init__()
      self.scale = tf.Variable(1.)

    def call(self, inputs):
      return inputs * self.scale


def custom_layer(x):
    x_sub = x - 0.9
    pos = K.relu(x_sub)
    pos_ids = K.greater(pos, 0)
    x_time = 0.025 * x[pos_ids] + 0.0125
    x_time_pad = K.zeros(16, dtype=float)
    #preprocessing.sequence.pad_sequences(x_time, maxlen=16, padding='post', dtype=float)
    return x_time_pad


epoch = 200
model_save_path = '../../data/models/rhthym_onset_dnn'

model = keras.models.load_model('{}/onset_model_epoch-{}.hdf5'.format(model_save_path,epoch))
model.add(layers.ScaleLayer())
model.add(layers.Dense(36, activation='relu'))
model.add(layers.Dense(72, activation='relu'))
model.add(layers.Dense(16, activation='relu'))

model.summary()

input_1 = np.load('../../data/feat/ref/51_rhy1_ref101559_grade4_spec.npy')

ss = model.predict(input_1[np.newaxis, :, :])
ss = 4