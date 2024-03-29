import sys
sys.path.append('../')

import os
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from callback.onsetCallBack import Onset_Callback
from model.models import Models
from dataset.onset import Dataset_Onset
from configuration import get_config
import tensorflow as tf

config = get_config()

if not os.path.exists(config.model_save_path):
    os.makedirs(config.model_save_path)
if not os.path.exists(config.log_save_path):
    os.makedirs(config.log_save_path)

# Mute excessively verbose Tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

param_str = 'onset'

###################
# Create datasets #
###################
train = Dataset_Onset(config=config)
valid = Dataset_Onset(config=config, is_test=True, audio_path_test=config.audio_path_test_per)

if config.model_name == 'LSTM_seq2seq':
    input_shape = [config.max_frm, config.num_feat]  # (batch_size, timespan, input_dim)
if config.model_name == 'CNN1D_RNN_seq2seq':
    input_shape = [config.max_frm, config.num_feat]
if config.model_name == 'CNN2D_RNN_seq2seq':
    input_shape = [config.max_frm, config.num_feat, 1]

train_generator = (batch for batch in train.yield_batches(config.batch_size))
valid_generator = (batch for batch in valid.yield_batches(config.batch_size))

################
# Define model #
################
model = Models(model_name=config.model_name,
                units=config.num_units,
                input_shape=input_shape,
                dropout=config.dropout)

model = model.encoder

opt = SGD() #opt = Adam(clipnorm=1.) #opt = SGD(clipnorm=1.)
loss_func = tf.keras.losses.MeanSquaredError() # loss_func = tf.keras.losses.MeanSquareError()
metrics_func = tf.keras.metrics.MeanSquaredError(name='mse') # metrics_func = [tf.keras.metrics.MeanAbsoluteError()]

model.compile(loss=loss_func, optimizer=opt, metrics=metrics_func)
model.summary()

#################
# Training Loop #
#################
model.fit(
    train_generator,
    steps_per_epoch=config.steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=200,
    epochs=(config.epochs + 1),
    verbose=False,
    #workers=multiprocessing.cpu_count(),
    #use_multiprocessing=True,
    callbacks=[
        # First generate custom n-shot classification metric
        Onset_Callback(valid),
        # Then log and checkpoint
        CSVLogger('{}/{}.csv'.format(config.log_save_path, param_str)),
        ModelCheckpoint(
            '{}/{}.hdf5'.format(config.model_save_path, param_str),
            monitor='mse',
            mode='min',
            save_best_only=True,
            verbose=True
        ),
        #ReduceLROnPlateau(
        #    monitor='mse',
        #    mode='min',
        #    verbose=True
        #)
    ]
)
