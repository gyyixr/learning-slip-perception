#!/usr/bin/env python2.7

"""
utils.py

General Utils for slip detection

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

# Standard Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # This is to suppress TF logs
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import math
import select
from datetime import datetime
from scipy.fftpack import rfft, fft

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, \
                            classification_report, confusion_matrix, cohen_kappa_score, \
                            accuracy_score, precision_recall_curve, roc_curve
from sklearn.metrics.pairwise import cosine_similarity

from nets import compiled_tcn, compiled_tcn_3D, tcn_full_summary, freq_model

# Get log directory
CWD = os.path.dirname(os.path.realpath(__file__))
logdir = CWD + "/../logs"


def fft_real(array ,axis=-1):
    assert len(np.shape(array)) > 0
    assert -len(np.shape(array)) <= axis < len(np.shape(array))
    assert isinstance(axis, int)

    return abs(fft(x=array, axis=axis))

class slip_detection_model:

    # TODO: fix slip Threshold
    def __init__(self, config, slip_thresh=0.5, num_features=6, num_classes=2):
        self.data_config = config['data']
        self.network_config = config['net']
        self.training_config = config['training']
        self.slip_thresh = slip_thresh

        # Extract data home
        self.data_home = self.data_config['data_home']

        if self.network_config['trained'] == True:
            # Load Model
            self.log_models_dir = self.network_config['model_dir']
            self.log_scalers = self.training_config['log_scaler_dir']
            self.log_best_model = self.network_config['best_model_path'] if 'best_model_path' in self.network_config else self.log_models_dir
            if self.network_config['use_best_model'] == True:
                self.__model = keras.models.load_model(self.log_best_model)
            else:
                self.__model = keras.models.load_model(self.log_models_dir)
            if self.network_config['type'] == 'tcn': tcn_full_summary(self.__model, expand_residual_blocks=True)
            elif self.network_config['type'] == 'tcn3D': tcn_full_summary(self.__model, expand_residual_blocks=True)
            elif self.network_config['type'] == 'freq_net': self.__model.summary()
            else: raise ValueError("Model type not supported: {}".format(self.network_config['type']))
        else:
            if self.network_config['type'] == 'tcn':
                # Create TCN self.__model
                output_layers = self.network_config['output_layers'][:]
                output_layers.append(num_classes)
                self.__model = compiled_tcn(return_sequences= self.network_config['return_sequences'],
                                    num_feat=          num_features,
                                    nb_filters=        self.network_config['nb_filters'],
                                    kernel_size=       self.network_config['kernel_size'],
                                    dilations=         self.network_config['dilations'],
                                    nb_stacks=         self.network_config['nb_stacks'],
                                    max_len=           self.data_config['series_len'],
                                    use_skip_connections=self.network_config['use_skip_connections'],
                                    regression=        self.training_config['regression'],
                                    dropout_rate=      self.training_config['dropout_rate'],
                                    activation=        self.network_config['activation'],
                                    opt=               self.training_config['opt'],
                                    use_batch_norm=    self.training_config['use_batch_norm'],
                                    use_layer_norm=    self.training_config['use_layer_norm'],
                                    lr=                self.training_config['lr'],
                                    kernel_initializer=self.training_config['kernel_initializer'],
                                    output_layers=     output_layers)
                self.log_models_dir = logdir + "/models/" + "TCN_" +  datetime.now().strftime("%Y%m%d-%H%M%S")
                tcn_full_summary(self.__model, expand_residual_blocks=True)
            elif self.network_config['type'] == 'tcn3D':
                output_layers = self.network_config['output_layers'][:]
                output_layers.append(num_classes)
                self.__model = compiled_tcn_3D(return_sequences=  self.network_config['return_sequences'],
                                        input_shape=       (2,3,1),
                                        nb_filters=        self.network_config['nb_filters'],
                                        kernel_size=       (3, 3, self.network_config['kernel_size']),
                                        dilations=         self.network_config['dilations'],
                                        nb_stacks=         self.network_config['nb_stacks'],
                                        max_len=           self.data_config['series_len'],
                                        use_skip_connections=self.network_config['use_skip_connections'],
                                        regression=        self.training_config['regression'],
                                        dropout_rate=      self.training_config['dropout_rate'],
                                        activation=        self.network_config['activation'],
                                        opt=               self.training_config['opt'],
                                        use_batch_norm=    self.training_config['use_batch_norm'],
                                        use_layer_norm=    self.training_config['use_layer_norm'],
                                        lr=                self.training_config['lr'],
                                        kernel_initializer=self.training_config['kernel_initializer'],
                                        output_layers=     output_layers)
                self.log_models_dir = logdir + "/models/" + "TCN3D_" +  datetime.now().strftime("%Y%m%d-%H%M%S")
                tcn_full_summary(self.__model, expand_residual_blocks=True)
            elif self.network_config['type'] == 'freq_net':
                output_layers = self.network_config['output_layers'][:]
                self.__model = freq_model(input_shape=(2,3,self.data_config['series_len']),
                                    num_classes=num_classes,
                                    cnn_filters_num=self.network_config['nb_filters'],
                                    dense_layer_num=output_layers,
                                    batch_norm=self.training_config['use_batch_norm'],
                                    dropout_rate=self.training_config['dropout_rate'],
                                    kernel_initializer=self.training_config['kernel_initializer'],
                                    padding=self.network_config['padding'],
                                    lr=self.training_config['lr'],
                                    activation = self.network_config['activation'])
                self.log_models_dir = logdir + "/models/" + "FREQ_" +  datetime.now().strftime("%Y%m%d-%H%M%S")
                self.__model.summary()
            else:
                raise ValueError("Model type not supported: {}".format(self.network_config['type']))
            # Create logging locations
            self.log_scalers = logdir + "/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            self.training_config['log_scaler_dir'] = self.log_scalers
            self.log_best_model = self.log_models_dir + "/best_model/"
            self.network_config['best_model_path'] = self.log_best_model

        # Create model directory for saving test results
        if not os.path.isdir(self.log_models_dir):
            os.mkdir(self.log_models_dir)

    def train(self, datagen_train, val_data):
        # Train Model
        if self.training_config['regression'] != True:
            print("Training data distribution: {}".format(datagen_train.get_class_nums()))
        epochs = int(self.training_config['epochs'])
        if epochs - self.training_config['epochs_complete'] > 0:
            # Create Tensorboard callback
            tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.log_scalers, profile_batch=0)
            # Create best self.__model callback (saves the best self.__model based on a metric)
            if self.training_config['regression'] == True:
                best_metric = 'val_loss'
                mode = 'min'
                min_delta = 0.0001
            else:
                best_metric = 'val_categorical_accuracy'
                mode = 'max'
                min_delta = 0.0001
            model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
                                                            filepath=self.log_best_model,
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
            if self.training_config['regression'] != True and self.training_config['class_weights'] == True:
                class_weights = np.sum(datagen_train.get_class_nums())/ \
                                (datagen_train.get_class_nums() * len(datagen_train.get_class_nums()))
                # class_weights = np.ones_like(datagen_train.class_ratios) - datagen_train.class_ratios
                # class_weights /= np.min(class_weights)
                class_weights = dict(enumerate(class_weights.tolist()))
                print("[LOG]: Class Weights being used {}".format(class_weights))

            ###
            #  THE MAIN LEARNING FUNCTION
            ###
            self.__model.fit(x=datagen_train,
                            verbose=self.training_config['verbosity'], #0: Suppress chatty output; use Tensorboard instead
                            epochs=epochs,
                            initial_epoch=self.training_config['epochs_complete'],
                            callbacks=[tensorboard_callback, model_checkpoint_callback],
                            class_weight=class_weights,
                            validation_data=(val_data[0], val_data[1]))
            ###
            # END OF MAIN LEARNING FUNCTION
            ###
        else:
            print("Network has been trained to {} epochs".format(self.training_config['epochs']))
            print("No more training Required")
        self.network_config['trained'] = True
        self.training_config['epochs_complete'] = epochs if epochs > self.training_config['epochs_complete'] else self.training_config['epochs_complete']
        self.training_config['epochs'] = 0

    def test_datagen(self, datagen):
        # test Model
        x_test, y_test, vel_test = datagen.get_all_batches()
        y_predict = self.test_data(data=x_test)
        x_test, y_test = datagen.get_inverse_transform(inputs= x_test,\
                                                       outputs=y_test)
        _, y_predict = datagen.get_inverse_transform(outputs=y_predict)

        return x_test, y_test, y_predict, vel_test

    def test_data(self, data):
        # test Model
        y = self.__model.predict(x=data, batch_size=self.data_config['batch_size'])
        return y

    def save_model(self):
        # Save Model
        self.network_config['model_dir'] = self.log_models_dir
        if self.network_config['save_last_model'] == True:
            self.__model.save(filepath=self.log_models_dir,
                        overwrite=True,
                        include_optimizer=True)
        else: # Ask to save the model and wait 30s
            print("Would you like to save the last trained model? (y/n)")
            # Wait for 30 seconds for a response
            i, o, e = select.select( [sys.stdin], [], [], 30)
            if (i):
                if sys.stdin.readline().strip() == 'y':
                    self.__model.save(filepath=self.log_models_dir,
                            overwrite=True,
                            include_optimizer=True)
                else:
                    print("\nWARNING: Last model not saved\n")
            else:
                print("\nWARNING: Last model not saved\n")

    def generate_and_save_test_report(self, datagen_val):

        x, y, y_predict, vel = self.test_datagen(datagen_val)

        # Evaluation Metrics
        print_string = self.__generate_classification_report(y, y_predict, "CLASSIFICATION REPORT")
        print(print_string)

        # Save Report
        with open(self.log_models_dir + "/classification_report_all_{}.txt".format( self.training_config['epochs_complete']), "w") \
            as text_file:
            text_file.write(print_string)

        # Plot results
        self.__plot_classification(vel[:, 0], vel[:, 1],
                            classes=(y_predict[:, 1] > self.slip_thresh),
                            name="classification plot (predicted)",
                            save_location=self.log_models_dir + "/classification_plot_all_predicted.png")
        self.__plot_classification(vel[:, 0], vel[:, 1],
                            classes=y.argmax(axis=1),
                            name="classification plot (actual)",
                            save_location=self.log_models_dir + "/classification_plot_all_actual.png")
        self.__plot_classification(vel[:, 0], vel[:, 1],
                            classes=y.argmax(axis=1) == (y_predict[:, 1] > self.slip_thresh),
                            name="classification plot (correct)",
                            save_location=self.log_models_dir + "/classification_plot_all_correct.png")
        if np.shape(y)[1] == 2:
            self.__plot_precision_recall_curve(y[:,0], y_predict[:,0], save_location=self.log_models_dir + "/PR_curve_no_slip.png")
            self.__plot_precision_recall_curve(y[:,1], y_predict[:,1], save_location=self.log_models_dir + "/PR_curve_slip.png")
            self.__plot_roc_curve(y[:,0], y_predict[:,0], save_location=self.log_models_dir + "/ROC_curve_no_slip.png")
            self.__plot_roc_curve(y[:,1], y_predict[:,1], save_location=self.log_models_dir + "/ROC_curve_slip.png")

        if 'materials' in self.data_config and 'test_material' in self.data_config \
            and self.data_config['test_material'] == True:
            dir_list_val = [self.data_home + self.data_config['test_dir']]
            materials = self.data_config['materials']
            for m in materials:
                exclude = self.data_config['test_data_exclude'][:]
                mats = materials[:]
                mats.remove(m)
                self.data_config['test_data_exclude'].extend(mats)

                # create datagenerator
                datagen_val.reset_data()
                datagen_val.load_data_from_dir(dir_list=dir_list_val,
                                            exclude=self.data_config['test_data_exclude'])
                datagen_val.load_data_attributes_from_config()
                if datagen_val.empty():
                    eprint("Empty datagenerator for material: {}".format(m))
                    continue

                # Test on validation data again
                x_m, y_m, y_predict_m, vel_m = self.test_datagen(datagen_val)
                print_string = self.__generate_classification_report(y_m, y_predict_m, "CLASSIFICATION REPORT {}".format(m))
                self.data_config['test_data_exclude'] = exclude[:]

                # Display on STDOUT
                print(print_string)

                # Save Report
                with open(self.log_models_dir + "/classification_report_{}_{}.txt".format(m, self.training_config['epochs_complete']), "w") \
                    as text_file:
                    text_file.write(print_string)

                # Plot results
                self.__plot_classification(vel_m[:, 0], vel_m[:, 1],
                                    classes=(y_predict_m[:, 1] > self.slip_thresh),
                                    name="classification plot (predicted)",
                                    save_location=self.log_models_dir + "/classification_plot_{}_predicted.png".format(m))
                self.__plot_classification(vel_m[:, 0], vel_m[:, 1],
                                    classes=y_m.argmax(axis=1),
                                    name="classification plot (actual)",
                                    save_location=self.log_models_dir + "/classification_plot_{}_actual.png".format(m))
                self.__plot_classification(vel_m[:, 0], vel_m[:, 1],
                                    classes=y_m.argmax(axis=1) == (y_predict_m[:, 1] > self.slip_thresh),
                                    name="classification plot (correct)",
                                    save_location=self.log_models_dir + "/classification_plot_{}_correct.png".format(m))

    def get_model_directory(self):
        return self.log_models_dir

    ## HELPER FUNCTIONS
    def __plot_classification(self, x, y,
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

    def __plot_precision_recall_curve(self, y_true, y_predict,
                                    name="precision vs recall",
                                    save_location=""):

        assert len(y_true) == len(y_predict)
        precision, recall, thresh = precision_recall_curve(y_true, y_predict)

        plot = plt.figure(figsize=(20, 20))
        plt.step(recall, precision, color='b', where='post')
        plt.plot([0,1],[0,1], color='r')
        plt.title(name)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xlim((0.5, 1.0))
        plt.ylim((0.5, 1.0))
        plt.axis('equal')
        plt.grid(True)

        for i,(x,y,z) in enumerate(zip(recall, precision, thresh)):
            if i % (len(thresh)/int(40)) == 0:
                label = "{:.2f}".format(z)
                plt.annotate(label, # this is the text
                            (x,y), # this is the point to label
                            textcoords="offset points", # how to position the text
                            xytext=(1, 0), # distance from text to points (x,y)
                            ha='left')

        if not save_location:
            plt.show(plot)
        else:
            assert ".png" in save_location
            plot.savefig(save_location, dpi=plot.dpi)

    def __plot_roc_curve(self, y_true, y_predict,
                        name="Recall Relationship",
                        save_location=""):

        assert len(y_true) == len(y_predict)
        fpr, tpr, thresh = roc_curve(y_true, y_predict)
        fpr = 1.0 - fpr

        plot = plt.figure(figsize=(10, 10))
        plt.step(fpr, tpr, color='b', where='post')
        plt.plot([0,1],[0,1], color='r')
        plt.title(name)
        plt.xlabel("Recall (Static)")
        plt.ylabel("Recall (Slip)")
        plt.axis('equal')
        plt.xlim((0.5, 1.0))
        plt.ylim((0.5, 1.0))
        plt.grid(True)

        for i,(x,y,z) in enumerate(zip(fpr, tpr, thresh)):
            if i % (len(thresh)/int(40)) == 0:
                label = "{:.2f}".format(z)
                plt.annotate(label, # this is the text
                            (x,y), # this is the point to label
                            textcoords="offset points", # how to position the text
                            xytext=(1, 0), # distance from text to points (x,y)
                            ha='left')

        if not save_location:
            plt.show(plot)
        else:
            assert ".png" in save_location
            plot.savefig(save_location, dpi=plot.dpi)

    def __generate_classification_report(self, y, y_predict,
                                       title = "EMPTY TITLE"):
        if len(y) == 0 or len(y_predict) == 0:
            eprint("Cannot generate classification results with empty arrrays")
            return ""

        assert np.shape(y) == np.shape(y_predict)

        print_string = title + "\n"
        class_matrix = classification_report(y.argmax(axis=1), (y_predict[:, 1] > self.slip_thresh), digits=4)
        cf_matrix = confusion_matrix(y.argmax(axis=1), (y_predict[:, 1] > self.slip_thresh))
        ck_score = cohen_kappa_score(y.argmax(axis=1), (y_predict[:, 1] > self.slip_thresh))
        class_accuracy = accuracy_score(y.argmax(axis=1), (y_predict[:, 1] > self.slip_thresh))
        print_string += "data: {}\n".format(self.data_config['data_home'])
        print_string += "exclude: {}\n".format(self.data_config['test_data_exclude'])
        print_string += "Accuracy: {}\n".format(class_accuracy)
        print_string += "This is the classification report: \n {}\n".format(class_matrix)
        print_string += "This is the confusion matrix: \n {}\n".format(cf_matrix)
        print_string += "This is the cohen Kappa score: \n {}".format(ck_score)

        return print_string

if __name__ == "__main__":
    a = np.random.random([10,32,21])
    print(a[0,0,:])
    print(fft_real(a, axis=-1)[0,0,:])