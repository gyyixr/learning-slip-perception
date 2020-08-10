#!/usr/bin/env python2.7

"""
DataGenerator.py

Keras datagenerator file for recorded takktile data

Developed at UTIAS, Toronto.

author: Abhinav Grover
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
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.io as sio

import tensorflow as tf
from DataLoader import takktile_dataloader, SPEED_THRESH

class takktile_datagenerator(tf.keras.utils.Sequence):

    def __init__(self, batch_size=32, shuffle=True, dataloaders=[], use_stream=False):
        """ Init function for takktile data generator

        Parameters
        ----------
        batch_size : int

        shuffle : bool

        dataloaders : takktile_dataloader

        use_stream : bool
            determines if each event occurance (False) is used to create inputs
            or only a series stream (True) of the same event occurance is
            considered a valid input
        """
        self.batch_size = batch_size
        self.dataloaders = dataloaders
        self.num_dl = len(dataloaders)
        self.shuffle = shuffle
        self.use_stream = use_stream

        # Reset and prepare data
        self.on_epoch_end()

    ###########################################
    #  PRIVATE FUNCTIONS
    ###########################################

    def __len__(self):
        if self.empty():
            return 0
        return int(self.num_n_slip_streams + self.num_n_slip_streams) / self.batch_size


    def __getitem__(self, index):
        if self.empty():
            return np.array([]), (np.array([]), np.array([]))

        if self.num_slip_streams + self.num_n_slip_streams > self.batch_size:
            # Uniformly choose between slip and non-slip
            X = np.empty([0, self.dataloaders[0].series_len, 6])
            Y_a = np.empty([0, 1])
            Y_b = np.empty([0, 1])
            for i in range(self.batch_size):
                x, y = [], []
                while len(x) == 0 or len(y) == 0:
                    idx = np.random.choice(self.dl_idx)
                    slip_ = np.random.choice([True, False])
                    array = self.slip_streams if slip_ else self.n_slip_streams
                    if array[idx]:
                        data_idx = array[idx].pop(0)
                        x, y = self.dataloaders[idx][data_idx]
                        if slip_:
                            self.num_slip_streams -= 1
                        else:
                            self.num_n_slip_streams -= 1
                    else:
                        continue
                X = np.append(X, np.expand_dims(x, axis=0), axis=0)
                Y_a = np.append(Y_a, np.reshape(y[0], [1,1]), axis=0)
                Y_b = np.append(Y_b, np.reshape(y[1], [1,1]), axis=0)

        return X, (Y_a, Y_b)


    ###########################################
    #  API FUNCTIONS
    ###########################################

    def empty(self):
        return self.num_dl <= 0

    def on_epoch_end(self):
        """ Created iterable list from dataloaders
        """
        if self.empty():
            print("WARNING: {}: dataloaders are not loaded yet, cannot process data".format(__file__))

        self.dl_idx = range(len(self.dataloaders))

        self.slip_streams = []
        self.n_slip_streams = []
        self.num_slip_streams = 0
        self.num_n_slip_streams = 0

        for dl in self.dataloaders:
            # Extract the slip and non-slip indices
            if self.use_stream:
                ss = dl.get_slip_stream_idx()
                n_ss = dl.get_no_slip_stream_idx()
            else:
                ss = dl.get_slip_idx()
                n_ss = dl.get_no_slip_idx()

            if self.shuffle:
                _ = np.random.shuffle(ss) if ss else None
                _ = np.random.shuffle(n_ss) if n_ss else None

            self.slip_streams.append(ss)
            self.n_slip_streams.append(n_ss)

            # Keep track of the number of each
            self.num_slip_streams += len(ss)
            self.num_n_slip_streams += len(n_ss)

    def load_data_from_dir(self, directory, series_len):
        """
        Recursive function to load data.mat files in directories with
        signature *takktile_* into a dataloader.

        Parameters
        ------------
        directory : str
            the root directory for the takktile data you wish to load
        series_len : int
            The length of the input time series
        """
        if not os.path.isdir(directory):
            eprint("\t\t {}: {} is not a directory".format(self.load_data_from_dir.__name__, directory))
            return

        dir_list = [directory]

        # Read Data from current directory
        while dir_list:
            current_dir = dir_list.pop(0)
            data_file = current_dir + "/data.mat"
            if os.path.isfile(data_file) and "takktile_" in current_dir:
                self.dataloaders.append(takktile_dataloader(current_dir, input_len=series_len, create_hist=False))

            # Find all child directories of takktile data and recursively load them
            data_dirs = [os.path.join(current_dir, o) for o in os.listdir(current_dir)
                    if os.path.isdir(os.path.join(current_dir, o)) and "takktile_" in o ]
            for d in data_dirs:
                dir_list.append(d)

        self.num_dl = len(self.dataloaders)
        # Reset and prepare data
        self.on_epoch_end()

if __name__ == "__main__":
    data_base = "/home/abhinavg/data/takktile/"
    dg = takktile_datagenerator()
    if dg.empty():
        print("The current Data generator is empty")
    dg.load_data_from_dir(directory=data_base, series_len=20)
    print("Num Batches: {}".format(len(dg)))