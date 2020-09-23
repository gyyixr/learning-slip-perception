#!/usr/bin/env python2.7

"""
test.py

Keras datagenerator file for recorded takktile data

Developed at UTIAS, Toronto.

author: Abhinav Grover

date: August 31, 2020
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
import sys
from datetime import datetime
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from nets import compiled_tcn, tcn_full_summary
from utils import takktile_datagenerator

#CONSTANTS
from utils import ALL_VALID, BOTH_SLIP, NO_SLIP, SLIP_TRANS, SLIP_ROT
MODELS_DIR = "./logs/models/"
PLOTS_DIR = "./logs/plots/"

def plot_prediction(true, predict,
                    axes=["true", "predicted"],
                    name="true vs predicted",
                    save_location=""):
    """
        Plotting function to plot true vs predicted value plot
        if save_location is empty, do not save and only show
    """
    assert len(true) == len(predict)

    plot = plt.figure(figsize=(10, 10))
    plt.scatter(true, predict)
    plt.title(name)
    plt.xlabel(axes[0])
    plt.ylabel(axes[1])
    plt.axis('equal')
    plt.grid(True)

    # Equal Plot
    line = np.linspace(np.min(true), np.max(true), 100)
    plt.plot(line, line, 'r')

    if not save_location:
        plt.show(plot)
    else:
        assert ".png" in save_location
        plot.savefig(save_location, dpi=plot.dpi)


def test_model(model, datagen):
    if not model:
        eprint("Cannot Evaluate without a model")
        raise ValueError("model cannot be none")

    # Train Model
    x_test, y_test = datagen.get_all_batches()
    bs = datagen.batch_size
    y_predict = model.predict(x=x_test, batch_size=bs)

    return x_test, y_test, y_predict


def test_translation(model_name, test_data_dir, batch_size=32, series_len=20):
    """
        Translation only training using translation dominant data which has been filtered to
        only include data points with high translation velocity and low rotation velocity
    """
    # Create datagenerator
    datagen_test = takktile_datagenerator(batch_size=batch_size,
                                           shuffle=True,
                                           data_mode=SLIP_TRANS,
                                           eval_data=False,
                                           transform='standard')
    # Load data into datagen
    dir_list = [test_data_dir]
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
    datagen_test.load_data_from_dir(dir_list=dir_list, series_len=series_len, rotation=False)

    # Test the data
    model = keras.models.load_model(MODELS_DIR + model_name)
    x, y, y_predict = test_model(model, datagen_test)

    # plot test data
    assert np.shape(y) == np.shape(y_predict)
    num_plots = np.shape(y)[1]
    for id in range(num_plots):
        plot_prediction(y[:, id], y_predict[:,id],
                        name="prediction plot for output dim {}".format(id),
                        save_location=PLOTS_DIR + model_name + "_{}.png".format(id))


def test_rotation(model_name, test_data_dir, batch_size=32, series_len=20):
    """
        Rotation only training using Rotation dominant data which has been filtered to
        only include data points with high Rotation velocity and low translation velocity
    """
    # Create datagenerator
    datagen_test = takktile_datagenerator(batch_size=batch_size,
                                           shuffle=True,
                                           data_mode=SLIP_ROT,
                                           eval_data=False,
                                           transform='standard')
    # Load data into datagen
    dir_list = [test_data_dir]
    while dir_list:
        current_dir = dir_list.pop(0)
        # Find all child directories of takktile data and recursively load them
        data_dirs = [os.path.join(current_dir, o) for o in os.listdir(current_dir)
                     if os.path.isdir(os.path.join(current_dir, o)) and
                    not ("translation" in o or "coupled" in o)]
        for d in data_dirs:
            dir_list.append(d)
        if all(["rotation" in d for d in dir_list]):
            break
    datagen_test.load_data_from_dir(dir_list=dir_list, series_len=series_len, translation=False)

    # Test the data
    model = keras.models.load_model(MODELS_DIR + model_name)
    x, y, y_predict = test_model(model, datagen_test)

    # plot test data
    assert np.shape(y) == np.shape(y_predict)
    num_plots = np.shape(y)[1]
    for id in range(num_plots):
        plot_prediction(y[:, id], y_predict[:,id],
                        name="prediction plot for output dim {}".format(id),
                        save_location=PLOTS_DIR + model_name + "_{}.png".format(id))

def test_all(model_name, test_data_dir, batch_size=32, series_len=20):
    """
        Test on all data
    """
    # Create datagenerator
    datagen_test = takktile_datagenerator(batch_size=batch_size,
                                           shuffle=True,
                                           data_mode=ALL_VALID,
                                           eval_data=False,
                                           transform='standard')
    # Load data into datagen
    dir_list = [test_data_dir]
    datagen_test.load_data_from_dir(dir_list=dir_list, series_len=series_len)

    # Test the data
    model = keras.models.load_model(MODELS_DIR + model_name)
    x, y, y_predict = test_model(model, datagen_test)

    # plot test data
    assert np.shape(y) == np.shape(y_predict)
    num_plots = np.shape(y)[1]
    for id in range(num_plots):
        plot_prediction(y[:, id], y_predict[:,id],
                        name="prediction plot for output dim {}".format(id),
                        save_location=PLOTS_DIR + model_name + "_{}.png".format(id))


if __name__ == "__main__":
    print("USAGE: train.py <model_name> <data_type>")
    args = sys.argv
    model_name = args[1]
    data_dir = "/home/abhinavg/data/takktile/data-v1/train/"
    mode = args[2]

    if mode == "all":
        test_all(model_name, data_dir, batch_size=32, series_len=100)
    elif mode == "trans":
        test_translation(model_name, data_dir, batch_size=32, series_len=100)
    elif mode == "rot":
        test_rotation(model_name, data_dir, batch_size=32, series_len=100)