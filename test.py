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
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow import keras

from nets import compiled_tcn, tcn_full_summary
from utils import takktile_datagenerator, load_yaml

#CONSTANTS
from utils import ALL_VALID, BOTH_SLIP, NO_SLIP, SLIP_TRANS, SLIP_ROT
CWD = os.path.dirname(os.path.realpath(__file__))
PLOTS_DIR = CWD + "/logs/plots"

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


def test_tcn(config):
    """
        Test tcn network based on the config provided
    """
    data_config = config['data']
    network_config = config['net']

    # Create datagenerator
    datagen_test = takktile_datagenerator(data_config)

    # Load data into datagen
    dir_list = [data_config['data_home'] + data_config['test_dir']]
    datagen_test.load_data_from_dir(dir_list=dir_list, exclude=data_config['test_data_exclude'])

    # Set data transform parameters
    _mean = data_config['data_transform']['mean']
    _std = data_config['data_transform']['std']
    _max = data_config['data_transform']['max']
    _min = data_config['data_transform']['min']
    mean_ = (np.array(_mean[0]), np.array(_mean[1]))
    std_ = (np.array(_std[0]), np.array(_std[1]))
    max_ = (np.array(_max[0]), np.array(_max[1]))
    min_ = (np.array(_min[0]), np.array(_min[1]))
    datagen_test.set_data_attributes(mean_, std_, max_, min_)

    # Test the data
    if network_config['trained'] == True:
        model = keras.models.load_model(network_config['model_dir'])
        x, y, y_predict = test_model(model, datagen_test)
    else:
        raise ValueError("Cannot test on untrained network")

    # plot test data
    assert np.shape(y) == np.shape(y_predict)
    num_plots = np.shape(y)[1]
    for id in range(num_plots):
        plot_prediction(y[:, id], y_predict[:,id],
                        name="prediction plot for output dim {}".format(id),
                        save_location= network_config['model_dir'] + "/true_vs_pred_{}.png".format(id))

    print("The mean squares error is: {}".format(mean_squared_error(y, y_predict)))
    print("The mean absolute error is: {}".format(mean_absolute_error(y, y_predict)))

if __name__ == "__main__":
    print("USAGE: train.py <config>")
    args = sys.argv
    config = load_yaml(args[1])

    if config['net']['type'] == 'tcn':
        test_tcn(config)