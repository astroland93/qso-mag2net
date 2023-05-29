# Generator used for model training
# (sorry documentation currently very sparse and  code 
# not too pretty but should do for now)

seed_value = 42


import keras
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.random.set_seed(seed_value)


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, data_hdf5, label_name_class, label_name_reg, batch_size=32, dim=(10958,), n_channels=1,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.data_hdf5 = data_hdf5
        self.label_name_class = label_name_class
        self.label_name_reg = label_name_reg
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __getitem__(self, index):
        'Generate one batch of data'    
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        # Generate data
        X, y = self.__data_generation(list_IDs_temp, self.data_hdf5, self.label_name_class, self.label_name_reg)
        
        return X, y
            
    def __data_generation(self, list_IDs_temp, data_hdf5, label_name_class, label_name_reg):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y_class = np.empty((self.batch_size), dtype=int)
        y_reg = np.empty((self.batch_size), dtype=float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = data_hdf5['flux'][ID]

        # Store class
            y_class[i] = data_hdf5[label_name_class][ID]
            y_reg[i] = data_hdf5[label_name_reg][ID]

        return X, {'out_class': y_class, 'out_reg': y_reg}