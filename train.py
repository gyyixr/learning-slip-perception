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
from utils import ALL_VALID, ALL_SLIP, NO_SLIP, SLIP_TRANS, SLIP_ROT


def train_tcn(datagen_train, val_data=()):
    # Get sample output
    test_x, test_y = datagen_train[0]

    # Create TCN model
    model = compiled_tcn(return_sequences=False,
                        num_feat=test_x.shape[2],
                        nb_filters=16,
                        kernel_size=8,
                        dilations=[2 ** i for i in range(9)],
                        nb_stacks=1,
                        max_len=test_x.shape[1],
                        use_skip_connections=True,
                        regression=True,
                        dropout_rate=0.1,
                        # use_batch_norm=True,
                        output_layers=[8 ,test_y.shape[1]])
    tcn_full_summary(model)

    # Create Tensorboard callback
    logdir = "./logs"
    log_scalers = logdir + "/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_models_dir = logdir + "/models/" + "TCN_" +  datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_scalers)

    # Train Model
    model.fit(x=datagen_train,
              verbose=1, #0: Suppress chatty output; use Tensorboard instead
              epochs=20,
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

def train_tcn_translation(batch_size=32, series_len=20):
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
                     if os.path.isdir(os.path.join(current_dir, o)) and
                    not ("rotation" in o or "coupled" in o)]
        for d in data_dirs:
            dir_list.append(d)
        if all(["translation" in d for d in dir_list]):
            break
    datagen_train.load_data_from_dir(dir_list=dir_list, series_len=series_len)
    train_tcn(datagen_train, datagen_train.evaluation_data())

def train_tcn_rotation(batch_size=32, series_len=20):
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
                    not ("translation" in o or "coupled" in o)]
        for d in data_dirs:
            dir_list.append(d)
        if all(["rotation" in d for d in dir_list]):
            break
    datagen_train.load_data_from_dir(dir_list=dir_list, series_len=series_len)
    train_tcn(datagen_train, datagen_train.evaluation_data())


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
    train_tcn(datagen_train, datagen_train.evaluation_data())

if __name__ == "__main__":
    train_tcn_all(batch_size=64, series_len=20)
