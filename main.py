#!/usr/bin/env python2.7

"""
main.py

Main file for slip detector

Developed at UTIAS, Toronto.

author: Abhinav Grover

date: Feb 10, 2021

External links:
    Batch Norm fix: https://github.com/tensorflow/tensorflow/issues/32477#issuecomment-574407290
"""

##################### Error printing
from __future__ import print_function
import sys

def eprint(*args, **kwargs):
    print("ERROR: {}: ".format(__file__))
    print(*args, file=sys.stderr, **kwargs)
#####################
import os

from utils import takktile_datagenerator, \
                  load_yaml, \
                  save_yaml, \
                  takktile_data_augment
from utils.utils import slip_detection_model

def create_train_val_datagen(config):
    data_config = config['data']
    network_config = config['net']
    training_config = config['training']

    # Extract data home
    data_home = data_config['data_home']

    # Create datagenerator Train
    datagen_train = takktile_datagenerator(config=data_config,
                                           augment=takktile_data_augment(data_config, noisy=True),
                                           balance = training_config['balance_data'] if 'balance_data' in training_config else False)

    # Load data into datagen
    dir_list_train = [data_home + data_config['train_dir']]
    datagen_train.load_data_from_dir(dir_list=dir_list_train,
                                     exclude=data_config['train_data_exclude'])

    # Create datagenerator Val
    datagen_val = takktile_datagenerator(config= data_config, augment=takktile_data_augment(None),
                                         balance= training_config['balance_data'] if 'balance_data' in training_config else False)

    # Load data into datagen
    dir_list_val = [data_home + data_config['test_dir']]
    datagen_val.load_data_from_dir(dir_list=dir_list_val,
                                   exclude=data_config['test_data_exclude'])

    # Load training tranformation
    if network_config['trained'] == True:
        datagen_train.load_data_attributes_from_config()
        datagen_val.load_data_attributes_from_config()
    else:
        datagen_train.load_data_attributes_to_config()
        datagen_val.load_data_attributes_from_config()

    return datagen_train, datagen_val


def train_slip_detection(config):

    # Create Data Generators
    datagen_train, datagen_val = create_train_val_datagen(config)

    val_data = datagen_val.get_all_batches()

    # Init Slip detection Network
    sd = slip_detection_model(config)

    # Train
    sd.train(datagen_train, val_data)

    # Generate Test Reports
    sd.generate_and_save_test_report(datagen_val)

    # Save Model
    sd.save_model()

    # Save config
    log_models_dir = sd.get_model_directory()
    save_yaml(config, log_models_dir + "/config.yaml")

    # Delete all variables
    del datagen_train, datagen_val

if __name__ == "__main__":
    print("Usage:  main.py <name of yaml config file>")
    config = load_yaml(sys.argv[1])

    train_slip_detection(config)