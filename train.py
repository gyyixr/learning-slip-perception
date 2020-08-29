#!/usr/bin/env python2.7

"""
train_tcn.py

Keras datagenerator file for recorded takktile data

Developed at UTIAS, Toronto.

author: Abhinav Grover

date: August 28, 2020
"""

##################### Error printing
from __future__ import print_function
import sys

def eprint(*args, **kwargs):
    print("ERROR: {}: ".format(__file__))
    print(*args, file=sys.stderr, **kwargs)
#####################
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

from nets import compiled_tcn, tcn_full_summary
from utils import takktile_datagenerator


#CONSTANTS
from utils import ALL_VALID, ALL_SLIP, NO_SLIP, SLIP_TRANS, SLIP_ROT

def train_tcn(batch_size=32, series_len=20):
    # Create datagenerator
    datagen_train = takktile_datagenerator(batch_size=batch_size,
                                           shuffle=True,
                                           data_mode=ALL_VALID)
    #Create Dataset
    # takktile_dataset = tf.data.Dataset.from_generator(takktile_datagenerator,
    #                                                   (()))

    # Load data into datagen
    data_dir = "/home/abhinavg/data/takktile/"
    datagen_train.load_data_from_dir(directory=data_dir, series_len=series_len)
    test_x, test_y = datagen_train[0]

    # Create TCN model
    model = compiled_tcn(return_sequences=False,
                        num_feat=test_x.shape[2],
                        nb_filters=24,
                        num_classes=0,
                        kernel_size=8,
                        dilations=[2 ** i for i in range(9)],
                        nb_stacks=1,
                        max_len=test_x.shape[1],
                        use_skip_connections=True,
                        regression=True,
                        dropout_rate=0,
                        output_len=test_y.shape[1])
    tcn_full_summary(model)

    # Create Tensorboard callback
    logdir = "./logs"
    log_scalers = logdir + "/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_models_file = logdir + "/models/" + "TCN_" +  datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_scalers)

    # Train Model
    model.fit(x=datagen_train,
              verbose=1, # Suppress chatty output; use Tensorboard instead
              epochs=100,
              callbacks=[tensorboard_callback],
              # MultiProcessing options
              max_queue_size=10,
              use_multiprocessing=False,
              workers=1,
              validation_data=datagen_train[len(datagen_train)])

    model.save(filepath=log_models_file,
               overwrite=True,
               include_optimizer=True)

if __name__ == "__main__":
    train_tcn(16, 50)
