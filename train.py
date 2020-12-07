#!/usr/bin/env python2.7

"""
train.py

Keras datagenerator file for recorded takktile data

Developed at UTIAS, Toronto.

author: Abhinav Grover

date: August 28, 2020

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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # This is to suppress TF logs
import numpy as np
import select
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, \
                            classification_report, confusion_matrix, cohen_kappa_score, \
                            accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

from nets import compiled_tcn, compiled_tcn_3D, tcn_full_summary
from utils import takktile_datagenerator, load_yaml, save_yaml, takktile_data_augment


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

def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )

    # norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    # cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    # cbar.ax.set_ylabel('Density')

    return ax

def plot_regression(true, predict,
                    axes=["true", "predicted"],
                    name="true vs predicted",
                    save_location=""):
    """
        Plotting function to plot true vs predicted value plot
        if save_location is empty, do not save and only show
    """
    assert len(true) == len(predict)

    plot = plt.figure(figsize=(20, 20))# Calculate the point density
    density_scatter(true, predict, ax=plt, bins=[10,10], marker='.', s=20, edgecolor='')
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

def plot_classification(x, y,
                        classes,
                        legend = [],
                        name="true vs predicted",
                        save_location=""):
    """
        Plotting function to plot true vs predicted value plot
        if save_location is empty, do not save and only show
    """
    class_list = range(np.max(classes)+1)
    if not legend:
        legend = map(str, class_list)

    assert len(x) == len(y) == len(classes)
    assert len(class_list) == len(legend)

    plot = plt.figure(figsize=(20, 20))
    cmap = cm.get_cmap('brg', len(class_list))
    for c in class_list:
        idx = np.argwhere(classes == c)
        plt.scatter(x[idx], y[idx], c=cmap(c), marker='.', s=20, label=legend[c], edgecolors='none') # marker='.')
    plt.title(name)
    plt.xlabel("x (m/s)")
    plt.ylabel("y (m/s)")
    plt.axis('equal')
    plt.grid(True)
    plt.legend(title="Classes")

    if not save_location:
        plt.show(plot)
    else:
        assert ".png" in save_location
        plot.savefig(save_location, dpi=plot.dpi)


def test_model(model, datagen):
    if not model:
        eprint("Cannot Evaluate without a model")
        raise ValueError("model cannot be none")

    # test Model
    x_test, y_test, vel_test = datagen.get_all_batches()
    bs = datagen.batch_size
    y_predict = model.predict(x=x_test, batch_size=bs)
    x_test, y_test = datagen.get_inverse_transform(inputs= x_test,\
                                                   outputs=y_test)
    _, y_predict = datagen.get_inverse_transform(outputs=y_predict)

    return x_test, y_test, y_predict, vel_test

def generate_regression_report(y, y_predict, data_config, title = "EMPTY TITLE"):
    if len(y) == 0 or len(y_predict) == 0:
        eprint("Cannot generate regression results with empty arrrays")
        return ""

    # Equality check
    assert np.shape(y) == np.shape(y_predict)

    print_string = title + "\n"
    print_string += "data: {}\n".format(data_config['data_home'])
    print_string += "exclude: {}\n".format(data_config['test_data_exclude'])
    if data_config['label_dimension'] == 'all' or data_config['label_dimension'] == 'translation':
        print_string += "The mean squares velocity error is: {} m^2/s^2\n".format(mean_squared_error(y[:, 0:2], y_predict[:, 0:2]))
        print_string += "The mean absolute velocity error is: {} m/s \n".format(mean_absolute_error(y[:, 0:2], y_predict[:, 0:2]))
        print_string += "Cosine similarity for velocity is: {}\n".format(mean_cosine_similarity(y[:, 0:2], y_predict[:, 0:2]))

    if data_config['label_dimension'] == 'all':
        print_string += "The mean squares rotation error is: {} rad^2/s^s\n".format(mean_squared_error(y[:, 2], y_predict[:, 2]))
        print_string += "The mean absolute rotation error is: {} rad/s\n".format(mean_absolute_error(y[:, 2], y_predict[:, 2]))
        # print_string += "Cosine Similarity for rotation error is: {} \n".format(mean_cosine_similarity(y[:, 2], y_predict[:, 2]))

    if data_config['label_dimension'] == 'rotation':
        print_string += "The mean squares rotation error is: {} rad^2/s^s\n".format(mean_squared_error(y, y_predict))
        print_string += "The mean absolute rotation error is: {} rad/s\n".format(mean_absolute_error(y, y_predict))
        # print_string += "Cosine Similarity for rotation error is: {} \n".format(mean_cosine_similarity(y[:, 2], y_predict[:, 2]))

    if data_config['label_dimension'] == 'x' or data_config['label_dimension'] == 'y':
        print_string += "The mean squares velocity error is: {} m^2/s^2\n".format(mean_squared_error(y, y_predict))
        print_string += "The mean absolute velocity error is: {} m/s \n".format(mean_absolute_error(y, y_predict))

    return print_string


def generate_classification_report(y, y_predict, data_config, title = "EMPTY TITLE"):
    if len(y) == 0 or len(y_predict) == 0:
        eprint("Cannot generate classification results with empty arrrays")
        return ""

    assert np.shape(y) == np.shape(y_predict)

    print_string = title + "\n"
    class_matrix = classification_report(y.argmax(axis=1), y_predict.argmax(axis=1))
    cf_matrix = confusion_matrix(y.argmax(axis=1), y_predict.argmax(axis=1))
    ck_score = cohen_kappa_score(y.argmax(axis=1), y_predict.argmax(axis=1))
    class_accuracy = accuracy_score(y.argmax(axis=1), y_predict.argmax(axis=1))
    print_string += "data: {}\n".format(data_config['data_home'])
    print_string += "exclude: {}\n".format(data_config['test_data_exclude'])
    print_string += "Accuracy: {}\n".format(class_accuracy)
    print_string += "This is the classification report: \n {}\n".format(class_matrix)
    print_string += "This is the confusion matrix: \n {}\n".format(cf_matrix)
    print_string += "This is the cohen Kappa score: \n {}".format(ck_score)

    return print_string

def train_net(config):
    print("\n\n****************TRAINING TCN********************\n\n")
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
    datagen_val = takktile_datagenerator(config=data_config, augment=takktile_data_augment(None))

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

    val_data = datagen_val.get_all_batches()

    # Get sample output
    test_x, test_y = datagen_train[0]
    print(np.shape(test_x))
    print(np.shape(test_y))

    if network_config['trained'] == True:
        # Load Model
        log_models_dir = network_config['model_dir']
        log_scalers = training_config['log_scaler_dir']
        log_best_model = network_config['best_model_path'] if 'best_model_path' in network_config else log_models_dir
        if network_config['use_best_model'] == True:
            model = keras.models.load_model(log_best_model)
        else:
            model = keras.models.load_model(log_models_dir)
    else:
        if network_config['type'] == 'tcn':
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
        elif network_config['type'] == 'tcn3D':
            output_layers = network_config['output_layers'][:]
            output_layers.append(test_y.shape[1])
            model = compiled_tcn_3D(return_sequences=  network_config['return_sequences'],
                                    input_shape=       (2,3,1),
                                    nb_filters=        network_config['nb_filters'],
                                    kernel_size=       (3, 3, network_config['kernel_size']),
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
            log_models_dir = logdir + "/models/" + "TCN3D_" +  datetime.now().strftime("%Y%m%d-%H%M%S")

        else:
            raise ValueError("Model type not supported: {}".format(network_config['type']))
        # Create logging locations
        log_scalers = logdir + "/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        training_config['log_scaler_dir'] = log_scalers
        log_best_model = log_models_dir + "/best_model/"
        network_config['best_model_path'] = log_best_model
    tcn_full_summary(model, expand_residual_blocks=True)

    # Train Model
    if training_config['regression'] != True:
        print("Training data distribution: {}".format(datagen_train.get_class_nums()))
    epochs = int(training_config['epochs'])
    if epochs - training_config['epochs_complete'] > 0:
        # Create Tensorboard callback
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_scalers)
        # Create best model callback (saves the best model based on a metric)
        if training_config['regression'] == True:
            best_metric = 'val_loss'
            mode = 'min'
            min_delta = 0.0001
        else:
            best_metric = 'val_categorical_accuracy'
            mode = 'max'
            min_delta = 0.0001
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
                                                        filepath=log_best_model,
                                                        save_weights_only=False,
                                                        monitor=best_metric,
                                                        mode=mode,
                                                        save_best_only=True)
        # Create Early stop callback
        es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                           mode='min',
                                           verbose=1,
                                           min_delta=min_delta,
                                           patience=50)

        # Compute class weights
        class_weights = []
        if training_config['regression'] != True and training_config['class_weights'] == True:
            class_weights = np.sum(datagen_train.get_class_nums())/ \
                            (datagen_train.get_class_nums() * len(datagen_train.get_class_nums()))
            # class_weights = np.ones_like(datagen_train.class_ratios) - datagen_train.class_ratios
            # class_weights /= np.min(class_weights)
            class_weights = dict(enumerate(class_weights.tolist()))
            print("[LOG]: Class Weights being used {}".format(class_weights))

        ###
        #  THE MAIN LEARNING FUNCTION
        ###
        model.fit(x=datagen_train,
                  verbose=training_config['verbosity'], #0: Suppress chatty output; use Tensorboard instead
                  epochs=epochs,
                  initial_epoch=training_config['epochs_complete'],
                  callbacks=[tensorboard_callback, model_checkpoint_callback],
                  class_weight=class_weights,
                  validation_data=(val_data[0], val_data[1]))
        ###
        # END OF MAIN LEARNING FUNCTION
        ###
    else:
        print("Network has been trained to {} epochs".format(training_config['epochs']))
        print("No more training Required")
    network_config['trained'] = True
    training_config['epochs_complete'] = epochs if epochs > training_config['epochs_complete'] else training_config['epochs_complete']
    training_config['epochs'] = 0

    # Test on validation data again
    if 'truncate_pressure' in data_config:
        tp = data_config['truncate_pressure']
        data_config['truncate_pressure'] = 0
    x, y, y_predict, vel = test_model(model, datagen_val)
    if 'truncate_pressure' in data_config:
        data_config['truncate_pressure'] = tp

    # Create model directory for saving test results
    if not os.path.isdir(log_models_dir):
        os.mkdir(log_models_dir)

    # Evaluation Metrics
    if training_config['regression'] == True:
        print_string = generate_regression_report(y, y_predict, data_config, "")
        print(print_string)

        # Save Report
        with open(log_models_dir + "/regression_report_{}.txt".format( training_config['epochs_complete']), "w") \
            as text_file:
            text_file.write(print_string)

        # plot prediction data
        assert np.shape(y) == np.shape(y_predict)
        num_plots = np.shape(y)[1]
        for id in range(num_plots):
            plot_regression(y[:, id], y_predict[:, id],
                            name="prediction plot for output dim {}".format(id),
                            save_location=log_models_dir + "/true_vs_pred_{}_{}.png"\
                                .format( training_config['epochs_complete'], id))

        if 'materials' in data_config and 'test_material' in data_config \
            and data_config['test_material'] == True:
            materials = data_config['materials']
            for m in materials:
                exclude = data_config['test_data_exclude'][:]
                mats = materials[:]
                mats.remove(m)
                data_config['test_data_exclude'].extend(mats)

                # create datageneratorx
                datagen_val.reset_data()
                datagen_val.load_data_from_dir(dir_list=dir_list_val,
                                               exclude=data_config['test_data_exclude'])
                datagen_val.load_data_attributes_from_config()
                # Test on validation data again
                if 'truncate_pressure' in data_config:
                    tp = data_config['truncate_pressure']
                    data_config['truncate_pressure'] = 0
                x_m, y_m, y_predict_m, vel_m = test_model(model, datagen_val)
                if 'truncate_pressure' in data_config:
                    data_config['truncate_pressure'] = tp

                print_string = generate_regression_report(y_m, y_predict_m, data_config, "REGRESSION REPORT {}".format(m))
                data_config['test_data_exclude'] = exclude[:]

                # Display on STDOUT
                print(print_string)

                # Save Report
                with open(log_models_dir + "/regression_report_{}_{}.txt".format(m, training_config['epochs_complete']), "w") \
                    as text_file:
                    text_file.write(print_string)

                # plot prediction data
                assert np.shape(y_m) == np.shape(y_predict_m)
                num_plots = np.shape(y_m)[1]
                for id in range(num_plots):
                    plot_regression(y_m[:, id], y_predict_m[:, id],
                                    name="prediction plot for output dim {}".format(id),
                                    save_location=log_models_dir + "/true_vs_pred_{}_{}_{}.png"\
                                        .format(m, training_config['epochs_complete'], id))

    else:
        print_string = generate_classification_report(y, y_predict, data_config, "CLASSIFICATION REPORT")
        print(print_string)

        # Save Report
        with open(log_models_dir + "/classification_report_all_{}.txt".format( training_config['epochs_complete']), "w") \
            as text_file:
            text_file.write(print_string)

        # Plot results
        plot_classification(vel[:, 0], vel[:, 1],
                            classes=y_predict.argmax(axis=1),
                            name="classification plot (predicted)",
                            save_location=log_models_dir + "/classification_plot_all_predicted.png")
        plot_classification(vel[:, 0], vel[:, 1],
                            classes=y.argmax(axis=1),
                            name="classification plot (actual)",
                            save_location=log_models_dir + "/classification_plot_all_actual.png")
        plot_classification(vel[:, 0], vel[:, 1],
                            classes=y.argmax(axis=1) == y_predict.argmax(axis=1),
                            name="classification plot (correct)",
                            save_location=log_models_dir + "/classification_plot_all_correct.png")

        if 'materials' in data_config and 'test_material' in data_config \
            and data_config['test_material'] == True:
            materials = data_config['materials']
            for m in materials:
                exclude = data_config['test_data_exclude'][:]
                mats = materials[:]
                mats.remove(m)
                data_config['test_data_exclude'].extend(mats)

                # create datagenerator
                datagen_val.reset_data()
                datagen_val.load_data_from_dir(dir_list=dir_list_val,
                                               exclude=data_config['test_data_exclude'])
                datagen_val.load_data_attributes_from_config()
                if datagen_val.empty():
                    eprint("Empty datagenerator for material: {}".format(m))
                    continue

                # Test on validation data again
                x_m, y_m, y_predict_m, vel_m = test_model(model, datagen_val)
                print_string = generate_classification_report(y_m, y_predict_m, data_config, "CLASSIFICATION REPORT {}".format(m))
                data_config['test_data_exclude'] = exclude[:]

                # Display on STDOUT
                print(print_string)

                # Save Report
                with open(log_models_dir + "/classification_report_{}_{}.txt".format(m, training_config['epochs_complete']), "w") \
                    as text_file:
                    text_file.write(print_string)

                # Plot results
                plot_classification(vel_m[:, 0], vel_m[:, 1],
                                    classes=y_predict_m.argmax(axis=1),
                                    name="classification plot (predicted)",
                                    save_location=log_models_dir + "/classification_plot_{}_predicted.png".format(m))
                plot_classification(vel_m[:, 0], vel_m[:, 1],
                                    classes=y_m.argmax(axis=1),
                                    name="classification plot (actual)",
                                    save_location=log_models_dir + "/classification_plot_{}_actual.png".format(m))
                plot_classification(vel_m[:, 0], vel_m[:, 1],
                                    classes=y_m.argmax(axis=1) == y_predict_m.argmax(axis=1),
                                    name="classification plot (correct)",
                                    save_location=log_models_dir + "/classification_plot_{}_correct.png".format(m))

    # Save Model
    network_config['model_dir'] = log_models_dir
    if network_config['save_last_model'] == True:
        model.save(filepath=log_models_dir,
                    overwrite=True,
                    include_optimizer=True)
    else: # Ask to save the model and wait 30s
        print("Would you like to save the last trained model? (y/n)")
        # Wait for 30 seconds for a response
        i, o, e = select.select( [sys.stdin], [], [], 30)
        if (i):
            if sys.stdin.readline().strip() == 'y':
                model.save(filepath=log_models_dir,
                        overwrite=True,
                        include_optimizer=True)
            else:
                print("\nWARNING: Last model not saved\n")
        else:
            print("\nWARNING: Last model not saved\n")

    # Preserve config
    save_yaml(config, log_models_dir + "/config.yaml")

    # Delete all variables
    del datagen_train, datagen_val, test_x, test_y, model

if __name__ == "__main__":
    print("Usage:  train.py <name of yaml config file>")
    config = load_yaml(sys.argv[1])

    if config['net']['type'] == 'tcn' or config['net']['type'] == 'tcn3D':
        train_net(config)