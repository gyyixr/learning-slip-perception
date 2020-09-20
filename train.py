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
import os
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

from nets import compiled_tcn, tcn_full_summary
from utils import takktile_datagenerator


#CONSTANTS
from utils import ALL_VALID, BOTH_SLIP, NO_SLIP, SLIP_TRANS, SLIP_ROT


def train_tcn(datagen_train, val_data=()):
    # Get sample output
    test_x, test_y = datagen_train[0]

    # Create TCN model
    model = compiled_tcn(return_sequences=False,
                        num_feat=test_x.shape[2],
                        nb_filters=24,
                        kernel_size=10,
                        dilations=[2 ** i for i in range(11)],
                        nb_stacks=1,
                        max_len=test_x.shape[1],
                        use_skip_connections=True,
                        regression=True,
                        dropout_rate=0.1,
                        # use_batch_norm=True,
                        output_layers=[16, 16, 8, test_y.shape[1]])
    tcn_full_summary(model)

    # Create Tensorboard callback
    logdir = "./logs"
    log_scalers = logdir + "/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_models_dir = logdir + "/models/" + "TCN_" +  datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_scalers)

    # Train Model
    model.fit(x=datagen_train,
              verbose=1, #0: Suppress chatty output; use Tensorboard instead
              epochs=50,
              callbacks=[tensorboard_callback],
              # MultiProcessing options
              max_queue_size=10,
              use_multiprocessing=False,
              workers=1,
              validation_data=val_data)

    model.save(filepath=log_models_dir,
               overwrite=True,
               include_optimizer=True)

def train_tcn_all(batch_size=32, series_len=20):
    """
        Train for both rotation and translation velocity using all the valid data
    """
    # Create datagenerator Train
    datagen_train = takktile_datagenerator(batch_size=batch_size,
                                           shuffle=True,
                                           data_mode=ALL_VALID,
                                           eval_data=False)

    # Load data into datagen
    dir_list = ["/home/abhinavg/data/takktile/train"]
    datagen_train.load_data_from_dir(dir_list=dir_list, series_len=series_len)

    # Create datagenerator Val
    datagen_val = takktile_datagenerator(batch_size=batch_size,
                                           shuffle=True,
                                           data_mode=ALL_VALID,
                                           eval_data=False)

    # Load data into datagen
    dir_list = ["/home/abhinavg/data/takktile/val"]
    datagen_val.load_data_from_dir(dir_list=dir_list, series_len=series_len)

    train_tcn(datagen_train, datagen_val.get_all_batches())

def train_tcn_translation(data_home, batch_size=32, series_len=20):
    """
        Translation only training using translation dominant data which has been filtered to
        only include data points with high translation velocity and low rotation velocity
    """
    # Create datagenerator
    datagen_train = takktile_datagenerator(batch_size=batch_size,
                                           shuffle=True,
                                           data_mode=SLIP_TRANS,
                                           eval_data=False,
                                           transform='minmax')
    # Load data into datagen
    dir_list = [data_home + "/train/"]
    while dir_list:
        current_dir = dir_list.pop(0)

        # Find all child directories of takktile data and recursively load them
        data_dirs = [os.path.join(current_dir, o) for o in os.listdir(current_dir)
                     if os.path.isdir(os.path.join(current_dir, o)) and
                    not ("rotation" in o or "coupled" in o)]
        for d in data_dirs:
            dir_list.append(d)
        if all(["translation" in d for d in dir_list]):
            break
    datagen_train.load_data_from_dir(dir_list=dir_list, series_len=series_len, rotation=False)

    # Create datagenerator Val
    datagen_val = takktile_datagenerator(batch_size=batch_size,
                                           shuffle=True,
                                           data_mode=SLIP_TRANS,
                                           eval_data=False,
                                           transform='minmax')

    # Load data into datagen
    dir_list = [data_home + "/val/"]
    datagen_val.load_data_from_dir(dir_list=dir_list, series_len=series_len, rotation=False)

    # Load training tranformation
    a,b,c,d = datagen_train.get_data_attributes()
    datagen_val.set_data_attributes(a,b,c,d)

    # Start Training
    train_tcn(datagen_train, datagen_val.get_all_batches())

def train_tcn_rotation(data_home, batch_size=32, series_len=20):
    """
        Rotation only training using rotation dominant data which has been filtered to
        only include data points with high rotation velocity and low translation velocity
    """
    # Create datagenerator
    datagen_train = takktile_datagenerator(batch_size=batch_size,
                                           shuffle=True,
                                           data_mode=SLIP_ROT,
                                           eval_data=True)
    # Load data into datagen
    dir_list = [data_home + "/train/"]
    while dir_list:
        current_dir = dir_list.pop(0)

        # Find all child directories of takktile data and recursively load them
        data_dirs = [os.path.join(current_dir, o) for o in os.listdir(current_dir)
                     if os.path.isdir(os.path.join(current_dir, o))and
                    not ("translation" in o or "coupled" in o)]
        for d in data_dirs:
            dir_list.append(d)
        if all(["rotation" in d for d in dir_list]):
            break
    datagen_train.load_data_from_dir(dir_list=dir_list, series_len=series_len, translation=False)

    # Create datagenerator Val
    datagen_val = takktile_datagenerator(batch_size=batch_size,
                                           shuffle=True,
                                           data_mode=SLIP_ROT,
                                           eval_data=False)

    # Load data into datagen
    dir_list = [data_home + "/val/"]
    datagen_val.load_data_from_dir(dir_list=dir_list, series_len=series_len, translation=False)

    train_tcn(datagen_train, datagen_val.get_all_batches())


def train_tcn_coupled(batch_size=32, series_len=20):
    # Create datagenerator
    datagen_train = takktile_datagenerator(batch_size=batch_size,
                                           shuffle=True,
                                           data_mode=ALL_VALID,
                                           eval_data=True)
    # Load data into datagen
    dir_list = ["/home/abhinavg/data/takktile/"]
    while dir_list:
        current_dir = dir_list.pop(0)

        # Find all child directories of takktile data and recursively load them
        data_dirs = [os.path.join(current_dir, o) for o in os.listdir(current_dir)
                     if os.path.isdir(os.path.join(current_dir, o))and
                    not ("rotation" in o or "translation" in o)]
        for d in data_dirs:
            dir_list.append(d)
        if all(["coupled" in d for d in dir_list]):
            break
    datagen_train.load_data_from_dir(dir_list=dir_list, series_len=series_len)

    # Create datagenerator Val
    datagen_val = takktile_datagenerator(batch_size=batch_size,
                                           shuffle=True,
                                           data_mode=ALL_VALID,
                                           eval_data=False)

    # Load data into datagen
    dir_list = ["/home/abhinavg/data/takktile/val"]
    datagen_val.load_data_from_dir(dir_list=dir_list, series_len=series_len)

    train_tcn(datagen_train, datagen_val.get_all_batches())

if __name__ == "__main__":
    train_tcn_translation(data_home = "/home/abhinavg/data/takktile/data-v1",
                          batch_size=32,
                          series_len=30)
