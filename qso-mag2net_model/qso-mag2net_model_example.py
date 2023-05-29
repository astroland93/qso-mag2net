# Example of the model training for qso-mag2net 
# (sorry documentation currently very sparse and  code 
# not too pretty but should do for now)

# random seed
seed_value = 42

import os
#setting gpu
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# setting random seeds
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.random.set_seed(seed_value)

# other imports
import h5py
import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, AveragePooling1D, Flatten
from keras.callbacks import CSVLogger, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# generator used for training
import sys
sys.path.append('../notebooks/')
from generators import generator_fiducial_model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True

session = InteractiveSession(config=config)
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# decay of the learning rate
def step_decay(epoch):
    if epoch >=0 and epoch < 20:
        lrate = 0.001
    if epoch >= 20 and epoch < 80:
        lrate = 0.0001
    if epoch >= 80 and epoch < 120:
        lrate = 0.00001
    if epoch >= 120 and epoch <= 150:
        lrate = 0.000001
    return lrate

sample = # sample path 

base_name = # save name of the model
model_path = # save path of the model
logger_path = # path to the log files to be created
checkpoint_path = # path to the checkpoints to be created
dim_1 = 6316


# generator initial setup
dim = (dim_1,1)
params_generator = {'dim': (dim_1,),
          'batch_size': 500,
          'n_classes': 1,
          'n_channels': 1,
          'shuffle': True}

# model setup 
X_input = keras.Input(dim)
X = Conv1D(64, 5, activation='relu')(X_input)
X = AveragePooling1D(3)(X)
X = Conv1D(128, 10, activation='relu')(X)
X = AveragePooling1D(3)(X)
X = Conv1D(256, 10, activation='relu')(X)
X = Conv1D(256, 10, activation='relu')(X)
X = Conv1D(256, 10, activation='relu')(X)
X = AveragePooling1D(3)(X)
X = Flatten()(X)
X = Dense(512, activation='relu')(X)
X = Dense(512, activation='relu')(X)
X_class_dense_out = Dense(1, activation='sigmoid', name='out_class')(X)
X_reg_dense_out = Dense(1, activation='relu', name='out_reg')(X)

lr_scheduler = LearningRateScheduler(step_decay)
csv_logger = CSVLogger(logger_path + base_name + '_history.csv', append=True)
opt = Adam(0.001)


model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path+base_name, 
                                            monitor='val_loss',
                                            mode='min',
                                            save_weights_only=False,
                                            save_best_only=True)

model = keras.Model(inputs = X_input, outputs = [X_class_dense_out,X_reg_dense_out])
model.compile(optimizer=opt, loss = {'out_reg':'MAE', 'out_class':'binary_crossentropy'}, metrics = {'out_reg':'MAE', 'out_class':'accuracy'}, loss_weights={'out_reg':1., 'out_class':300})

print(model.summary())

sample_size = len(sample['flux'])

# training validation split and further generator setup
list_IDs_training = random.sample(range(sample_size), int(sample_size*0.8))
list_IDs_validation = range(0, sample_size)

list_IDs_validation = np.setdiff1d(list_IDs_validation, list_IDs_training)

training_generator = generator_WLfull.DataGenerator(list_IDs_training, sample, 'absorber_true', 'cent_WL_2796',  **params_generator)
validation_generator = generator_WLfull.DataGenerator(list_IDs_validation, sample, 'absorber_true', 'cent_WL_2796', **params_generator)

# model fitting
history = model.fit(training_generator, validation_data = validation_generator, epochs=150, verbose=1, shuffle=True, callbacks=[lr_scheduler, csv_logger, model_checkpoint_callback])

#saving model
model.save(model_path+base_name+'_model')
