#!/usr/bin/env python2.7

"""
DataGenerator.py

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

# Standard Imports
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.io as sio

import tensorflow as tf
from DataLoader import takktile_dataloader

# CONSTANTS
ALL_VALID = 1
ALL_SLIP = 2
NO_SLIP = 3
SLIP_TRANS = 4
SLIP_ROT = 5


class takktile_datagenerator(tf.keras.utils.Sequence):
    """
    Keras datagenerator for takktile slip detection data

    Note: use load_data_from_dir function to populate data
    """

    def __init__(self, batch_size=32,
                       shuffle=True,
                       dataloaders=[],
                       data_mode=ALL_VALID,
                       eval_data=False):
        """ Init function for takktile data generator

        Parameters
        ----------
        batch_size : int

        shuffle : bool

        dataloaders : takktile_dataloader

        data_mode: str
            can only take a few options:
            ALL_VALID
            ALL_SLIP
            NO_SLIP
            SLIP_TRANS
            SLIP_ROT

        eval_data: bool

        """
        self.batch_size = batch_size
        self.dataloaders = dataloaders
        self.num_dl = len(dataloaders)
        self.shuffle = shuffle
        self.data_mode = data_mode
        self.series_len = 0 if self.num_dl == 0 else self.dataloaders[0].series_len
        self.create_eval_data = eval_data
        self.eval_len = 0

        # Reset and prepare data
        self.on_epoch_end()

    ###########################################
    #  API FUNCTIONS
    ###########################################

    def empty(self):
        return self.num_dl <= 0

    def load_data_from_dir(self,
                           dir_list=[],
                           series_len=20,
                           translation=True,
                           rotation=True):
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
        for directory in dir_list:
            if not os.path.isdir(directory):
                eprint("\t\t {}: {} is not a directory".format(self.load_data_from_dir.__name__, directory))
                return

        self.series_len = series_len

        # Read Data from current directory
        while dir_list:
            current_dir = dir_list.pop(0)
            data_file = current_dir + "/data.mat"
            if os.path.isfile(data_file) and "takktile_" in current_dir:
                self.dataloaders.append(takktile_dataloader(data_dir=current_dir,
                                                            input_len=series_len,
                                                            create_hist=False,
                                                            rotation=rotation,
                                                            translation=translation))
                # self.dataloaders[-1].save_slip_hist(directory=current_dir)

            # Find all child directories of takktile data and recursively load them
            data_dirs = [os.path.join(current_dir, o) for o in os.listdir(current_dir)
                    if os.path.isdir(os.path.join(current_dir, o))]
            for d in data_dirs:
                dir_list.append(d)

        self.num_dl = len(self.dataloaders)
        # Reset and prepare data
        self.on_epoch_end()
        if self.create_eval_data:
            self.eval_len = (self.__len__())//10
            self.create_eval_data = False

    def on_epoch_end(self):
        """ Created iterable list from dataloaders
        """
        if self.empty():
            print("WARNING: {}: dataloaders are not loaded yet, cannot process data".format(__file__))

        self.dl_idx = range(len(self.dataloaders))
        self.dl_data_idx = []
        for idx in self.dl_idx:
            # Extract data indices
            data_idx_list = self.__get_data_idx(idx)
            if self.shuffle: # Shuffle dataloader data list
                np.random.shuffle(data_idx_list)
            # Store Data indices
            self.dl_data_idx.append(data_idx_list)

        if self.shuffle: # Shuffle dataloader list
            np.random.shuffle(self.dl_idx)

    def evaluation_data(self):
        if not self.create_eval_data:
            eprint("No eval Data available")
        eval_len = self.eval_len
        self.eval_len = 0
        X, Y = self.__get_batches([self.__len__() -i-1 for i in range(eval_len)])
        self.eval_len = eval_len
        return X, Y

    def get_all_batches(self):
        return self.__get_batches(range(self.__len__()))

    ###########################################
    #  PRIVATE FUNCTIONS
    ###########################################


    def __len__(self):
        if self.empty():
            return 0
        num = 0
        for dl_idx in self.dl_data_idx:
            num += len(dl_idx)
        real_len = int(num) / self.batch_size
        return real_len - self.eval_len

    def __get_batches(self, batches=[]):
        X_ = np.empty([0])
        Y_ = np.empty([0])
        for i in batches:
            X, Y = self.__getitem__(i)
            X_ = X if X_.size == 0 else np.append(X_, X, axis=0)
            Y_ = Y if Y_.size == 0 else np.append(Y_, Y, axis=0)
        if X_.size == 0 or Y_.size == 0:
            return
        return X_, Y_

    def __getitem__(self, batch_index):
        if self.empty() or batch_index >= self.__len__() or batch_index < 0:
            eprint("Index out of bounds")
            raise ValueError("Index {} out of bounds for length {}".format(batch_index, self.__len__()))

        # Fetching data from [index*bs] -> [(index+1)*bs]
        indices = range(batch_index*self.batch_size, (batch_index+1)*self.batch_size)
        x_, y_ = self.dataloaders[0][0]
        X = np.empty([0, self.series_len, np.size(x_[0])])
        Y = np.empty([0, np.size(y_)])
        for i in indices:
            x = None
            y = None
            dl_id = 0
            for dl_id in self.dl_idx:
                # Find the bin that i belongs to
                if i >= len(self.dl_data_idx[dl_id]):
                    i -= len(self.dl_data_idx[dl_id])
                else:
                    break
            x, y = self.dataloaders[dl_id][self.dl_data_idx[dl_id][i]]
            X = np.append(X, np.expand_dims(np.array(x), axis=0), axis=0)
            Y = np.append(Y, np.expand_dims(np.array(y, ndmin=1), axis=0), axis=0)

        return X, Y


    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item

    def __get_data_idx(self, dl_id):
        if self.data_mode == ALL_VALID:
            return self.dataloaders[dl_id].get_valid_idx()
        elif self.data_mode == ALL_SLIP:
            return self.dataloaders[dl_id].get_slip_n_rot_idx()
        elif self.data_mode == NO_SLIP:
            return self.dataloaders[dl_id].get_no_slip_idx()
        elif self.data_mode == SLIP_TRANS:
            return self.dataloaders[dl_id].get_slip_idx()
        elif self.data_mode == SLIP_ROT:
            return self.dataloaders[dl_id].get_rot_idx()
        else:
            eprint("Unrecognised data mode: {}".format(self.data_mode))
            raise ValueError("Unrecognised data mode")

if __name__ == "__main__":
    data_base = "/home/abhinavg/data/takktile/train"
    dg = takktile_datagenerator()
    if dg.empty():
        print("The current Data generator is empty")
    dg.load_data_from_dir(dir_list=[data_base], series_len=20)
    print("Num Batches: {}".format(len(dg)))
    print("First Batch Comparison")
    for i in range(len(dg)):
        print(i)
        print(np.all(dg[i][0] == dg[i][0]))
        print(np.all(dg[i][1] == dg[i][1]))
