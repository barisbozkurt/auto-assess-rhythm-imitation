import sys
sys.path.append('../')

import os
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from callback.siameseCallBack import Siamese_Callback
from model.models import Models
from model.models import Siamese
from dataset.siamese import Dataset_Siamese
from configuration import get_config
import tensorflow as tf

config = get_config()

# Mute excessively verbose Tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

param_str = 'rhtyhm_siamese'

###################
# Create datasets #
###################
train = Dataset_Siamese(config.audio_path_trn_per, config.audio_path_ref, config.model_name)
valid = Dataset_Siamese(config.audio_path_test_per, config.audio_path_ref, config.model_name)
if config.model_name == 'CNN1D_RNN_pretrained':
    input_shape = [config.max_frm, config.num_feat]
if config.model_name == 'CNN1D_RNN':
    input_shape = [config.max_frm, config.num_feat]
if config.model_name == 'CNN2D':
    input_shape = [config.max_frm, config.num_feat, 1]
if config.model_name == 'CNN1D':
    input_shape = [config.max_frm, config.num_feat]
if config.model_name == 'FF':
    input_shape = [config.num_feat]

train_generator = (batch for batch in train.yield_batches(config.batch_size))
valid_generator = (batch for batch in valid.yield_batches(config.batch_size))

################
# Define model #
################
model = Models(model_name=config.model_name,
               embedding_dimension=config.embedding_dimension,
               filters=config.num_filter,
               input_shape=input_shape,
               dropout=config.dropout)
encoder = model.encoder

model_siamese = Siamese(encoder=encoder, input_shape=input_shape)

siamese = model_siamese.siamese

opt = SGD() #opt = Adam(clipnorm=1.)#opt = SGD(clipnorm=1.)
loss_func = tf.keras.losses.MeanSquaredError() # loss_func = tf.keras.losses.MeanSquareError()
metrics_func = [tf.keras.metrics.MeanSquaredError(name='reg_loss')] # metrics_func = [tf.keras.metrics.MeanAbsoluteError()]

siamese.compile(loss=loss_func, optimizer=opt, metrics=metrics_func)
#plot_model(siamese, show_shapes=True, to_file='plots/siamese.png')
siamese.summary()

#################
# Training Loop #
#################
siamese.fit(
    train_generator,
    steps_per_epoch=config.steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=20,
    epochs=(config.epochs+1),
    verbose=True,
    #workers=multiprocessing.cpu_count(),
    #use_multiprocessing=True,
    callbacks=[
        # First generate custom n-shot classification metric
        Siamese_Callback(valid),
        # Then log and checkpoint
        CSVLogger('../../data/logs/{}.csv'.format(param_str)),
        ModelCheckpoint(
            '../../data/models/{}.hdf5'.format(param_str),
            monitor='reg_loss',
            mode='min',
            save_best_only=True,
            verbose=True
        ),
        #ReduceLROnPlateau(
        #    monitor='reg_loss',
        #    mode='max',
        #    verbose=True
        #)
    ]
)
