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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # This is to suppress TF logs
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

from nets import compiled_tcn, tcn_full_summary
from utils import takktile_datagenerator, load_yaml, save_yaml


#CONSTANTS
from utils import ALL_VALID, BOTH_SLIP, NO_SLIP, SLIP_TRANS, SLIP_ROT
CWD = os.path.dirname(os.path.realpath(__file__))
logdir = CWD + "/logs"

def train_tcn(config):
    data_config = config['data']
    network_config = config['net']
    training_config = config['training']

    # Extract data home
    data_home = data_config['data_home']

    # Create datagenerator Train
    datagen_train = takktile_datagenerator(data_config)

    # Load data into datagen
    dir_list = [data_home + data_config['train_dir']]
    datagen_train.load_data_from_dir(dir_list=dir_list,
                                     exclude=data_config['train_data_exclude'])

    # Create datagenerator Val
    datagen_val = takktile_datagenerator(data_config)

    # Load data into datagen
    dir_list = [data_home + data_config['test_dir']]
    datagen_val.load_data_from_dir(dir_list=dir_list,
                                   exclude=data_config['test_data_exclude'])

    # Load training tranformation
    mean, std, max_, min_ = datagen_train.get_data_attributes()
    datagen_val.set_data_attributes(mean, std, max_, min_)
    data_config['data_transform']['mean'] = (mean[0].tolist(), mean[1].tolist())
    data_config['data_transform']['std'] = (std[0].tolist(), std[1].tolist())
    data_config['data_transform']['max'] = (max_[0].tolist(), max_[1].tolist())
    data_config['data_transform']['min'] = (min_[0].tolist(), min_[1].tolist())

    # Get sample output
    test_x, test_y = datagen_train[0]

    if network_config['trained'] == True:
        log_models_dir = network_config['model_dir']
        model = keras.models.load_model(log_models_dir)
    else:
        # Create TCN model
        output_layers = network_config['output_layers']
        output_layers.append(test_y.shape[1])
        model = compiled_tcn(return_sequences= network_config['return_sequences'],
                            num_feat=          test_x.shape[2],
                            nb_filters=        network_config['nb_filters'],
                            kernel_size=       network_config['kernel_size'],
                            dilations=         network_config['dilations'],
                            nb_stacks=         network_config['nb_stacks'],
                            max_len=           test_x.shape[1],
                            use_skip_connections=network_config['use_skip_connections'],
                            regression=        training_config['regression'],
                            dropout_rate=      training_config['dropout_rate'],
                            activation=        network_config['activation'],
                            opt=               training_config['opt'],
                            use_batch_norm=    training_config['use_batch_norm'],
                            use_layer_norm=    training_config['use_layer_norm'],
                            lr=                training_config['lr'],
                            kernel_initializer=training_config['kernel_initializer'],
                            output_layers=     output_layers)
    tcn_full_summary(model)
    log_models_dir = logdir + "/models/" + "TCN_" +  datetime.now().strftime("%Y%m%d-%H%M%S")
    network_config['model_dir'] = log_models_dir

    # Create Tensorboard callback
    log_scalers = logdir + "/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_scalers)

    # Train Model
    epochs = int(training_config['epochs'] - training_config['epochs_complete'])
    if epochs > 0:
        model.fit(x=datagen_train,
                  verbose=training_config['verbosity'], #0: Suppress chatty output; use Tensorboard instead
                  epochs=epochs,
                  callbacks=[tensorboard_callback],
                  validation_data=datagen_val.get_all_batches())
    else:
        print("Network has been trained to {} epochs".format(training_config['epochs']))
        print("No more training Required")
    network_config['trained'] = True
    training_config['epochs_complete'] = training_config['epochs']

    # Save Model
    model.save(filepath=log_models_dir,
               overwrite=True,
               include_optimizer=True)

    # Preserve config
    save_yaml(config, log_models_dir + "/config.yaml")

    # Delete all variables
    del datagen_train, datagen_val, test_x, test_y, model

if __name__ == "__main__":
    print("Usage:  train.py <name of yaml config file>")
    config = load_yaml(sys.argv[1])

    if config['net']['type'] == 'tcn':
        train_tcn(config)