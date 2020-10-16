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
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

from nets import compiled_tcn, tcn_full_summary
from utils import takktile_datagenerator, load_yaml, save_yaml


#CONSTANTS
from utils import ALL_VALID, BOTH_SLIP, NO_SLIP, SLIP_TRANS, SLIP_ROT
CWD = os.path.dirname(os.path.realpath(__file__))
logdir = CWD + "/logs"

def mean_cosine_similarity(X, Y):
    if not np.shape(X) == np.shape(Y):
        eprint("Array shapes are different: {} | {}".format(np.shape(X), np.shape(Y)))
        return 0.0

    total_sim = 0.0
    for i in range(np.shape(X)[0]//2):
        total_sim += np.mean(np.diag(cosine_similarity(X[2*i : 2*i+2, :], Y[2*i : 2*i+2, :])))

    return total_sim/(np.shape(X)[0]//2)



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
    _, y_test = datagen.get_inverse_transform(outputs=y_test)
    _, y_predict = datagen.get_inverse_transform(outputs=y_predict)

    return x_test, y_test, y_predict

def train_tcn(config):
    data_config = config['data']
    network_config = config['net']
    training_config = config['training']

    # Extract data home
    data_home = data_config['data_home']

    # Create datagenerator Train
    datagen_train = takktile_datagenerator(config=data_config)

    # Load data into datagen
    dir_list_train = [data_home + data_config['train_dir']]
    datagen_train.load_data_from_dir(dir_list=dir_list_train,
                                     exclude=data_config['train_data_exclude'])

    # Create datagenerator Val
    datagen_val = takktile_datagenerator(config=data_config)

    # Load data into datagen
    dir_list_val = [data_home + data_config['test_dir']]
    datagen_val.load_data_from_dir(dir_list=dir_list_val,
                                   exclude=data_config['test_data_exclude'])

    # Load training tranformation
    mean, std, max_, min_ = datagen_train.get_data_attributes()
    datagen_val.set_data_attributes(mean, std, max_, min_)
    data_config['data_transform']['mean'] = (mean[0].tolist(), mean[1].tolist())
    data_config['data_transform']['std'] = (std[0].tolist(), std[1].tolist())
    data_config['data_transform']['max'] = (max_[0].tolist(), max_[1].tolist())
    data_config['data_transform']['min'] = (min_[0].tolist(), min_[1].tolist())

    val_data = datagen_val.get_all_batches()

    # Get sample output
    test_x, test_y = datagen_train[0]
    print(np.shape(test_x))
    print(np.shape(test_y))

    if network_config['trained'] == True:
        # Load Model
        log_models_dir = network_config['model_dir']
        log_scalers = training_config['log_scaler_dir']
        model = keras.models.load_model(log_models_dir)
    else:
        # Create TCN model
        output_layers = network_config['output_layers'][:]
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
        log_models_dir = logdir + "/models/" + "TCN_" +  datetime.now().strftime("%Y%m%d-%H%M%S")
        log_scalers = logdir + "/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        network_config['model_dir'] = log_models_dir
        training_config['log_scaler_dir'] = log_scalers
    tcn_full_summary(model)

    # Create Tensorboard callback
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_scalers)

    # Train Model
    epochs = int(training_config['epochs'])
    if epochs > 0:
        model.fit(x=datagen_train,
                  verbose=training_config['verbosity'], #0: Suppress chatty output; use Tensorboard instead
                  epochs=epochs,
                  callbacks=[tensorboard_callback],
                  validation_data=(val_data[0], val_data[1]))
    else:
        print("Network has been trained to {} epochs".format(training_config['epochs']))
        print("No more training Required")
    network_config['trained'] = True
    training_config['epochs_complete'] += epochs

    # test on validation data again
    x, y, y_predict = test_model(model, datagen_val)

    # Save Model
    model.save(filepath=log_models_dir,
               overwrite=True,
               include_optimizer=True)

    # Metrics
    if data_config['label_type'] == 'value':
        if data_config['label_dimension'] == 'all' or data_config['label_dimension'] == 'translation':
            print("The mean squares velocity error is: {} m^2/s^2".format(mean_squared_error(y[:, 0:2], y_predict[:, 0:2])))
            print("The mean absolute velocity error is: {} m/s ".format(mean_absolute_error(y[:, 0:2], y_predict[:, 0:2])))
            print("Cosine similarity for velocity is: {}".format(mean_cosine_similarity(y[:, 0:2], y_predict[:, 0:2])))

        if data_config['label_dimension'] == 'all':
            print("The mean squares rotation error is: {} rad^2/s^s".format(mean_squared_error(y[:, 2], y_predict[:, 2])))
            print("The mean absolute rotation error is: {} rad/s".format(mean_absolute_error(y[:, 2], y_predict[:, 2])))
            # print("Cosine Similarity for rotation error is: {} ".format(mean_cosine_similarity(y[:, 2], y_predict[:, 2])))

        elif data_config['label_dimension'] == 'rotation':
            print("The mean squares rotation error is: {} rad^2/s^s".format(mean_squared_error(y, y_predict)))
            print("The mean absolute rotation error is: {} rad/s".format(mean_absolute_error(y, y_predict)))
            # print("Cosine Similarity for rotation error is: {} ".format(mean_cosine_similarity(y[:, 2], y_predict[:, 2])))

        elif data_config['label_dimension'] == 'x' or data_config['label_dimension'] == 'y':
            print("The mean squares velocity error is: {} m^2/s^2".format(mean_squared_error(y, y_predict)))
            print("The mean absolute velocity error is: {} m/s ".format(mean_absolute_error(y, y_predict)))
    else:
        class_matrix = classification_report(y.argmax(axis=1), y_predict.argmax(axis=1))
        cf_matrix = confusion_matrix(y.argmax(axis=1), y_predict.argmax(axis=1))
        print("This is the classification report: \n {}".format(class_matrix))
        print("This is the confusion matrix: \n P0 |  P1 \n {}".format(cf_matrix))
        with open(log_models_dir + "/classification_report_{}.txt".format( training_config['epochs_complete']), "w") \
            as text_file:
            text_file.write("This is the classification report: \n {}".format(class_matrix))
            text_file.write("This is the confusion matrix: \n P0 |  P1 \n{}".format(cf_matrix))

    if data_config['label_type'] == 'value':
        # plot test data
        assert np.shape(y) == np.shape(y_predict)
        num_plots = np.shape(y)[1]
        for id in range(num_plots):
            plot_prediction(y[:, id], y_predict[:, id],
                            name="prediction plot for output dim {}".format(id),
                            save_location=log_models_dir + "/true_vs_pred_{}_{}.png"\
                                .format( training_config['epochs_complete'], id))

    # Preserve config
    save_yaml(config, log_models_dir + "/config.yaml")

    # Delete all variables
    del datagen_train, datagen_val, test_x, test_y, model

if __name__ == "__main__":
    print("Usage:  train.py <name of yaml config file>")
    config = load_yaml(sys.argv[1])

    if config['net']['type'] == 'tcn':
        train_tcn(config)