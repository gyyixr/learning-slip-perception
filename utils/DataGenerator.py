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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import tensorflow as tf
from DataLoader import takktile_dataloader

# CONSTANTS
ALL_VALID = 1
BOTH_SLIP = 2
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
                       eval_data=False,
                       transform = None):
        """ Init function for takktile data generator

        Parameters
        ----------
        batch_size : int

        shuffle : bool

        dataloaders : takktile_dataloader

        data_mode: str
            can only take a few options:
            ALL_VALID - all valid data
            BOTH_SLIP - data containing both rotation and translation slip
            NO_SLIP - no slip in data
            SLIP_TRANS - only translation slip in data
            SLIP_ROT - only rotation slip in data

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
        self.transform_type = transform

        if transform:
            assert transform == 'standard' or transform == 'minmax'

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
        translation: bool
            indicated whether translation speed should be included in the output or not
        rotation: bool
            indicated whether rotation speed should be included in the output or not
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

        if self.transform_type:
            self.__calculate_data_transforms()

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

    def set_data_attributes(self, means, stds, maxs, mins):
        """
        Setter funtion for data attributes
        means, stds, maxs, and mins should include ones for both inputs
        and outputs
        """
        # Assuming all inputs have the signature (in, out)
        if not self.transform_type:
            raise ValueError

        # Create Min Max scaler
        assert len(mins) == len(maxs) == 2
        self.min_in, self.min_out = mins
        self.max_in, self.max_out = maxs

        self.mm_scaler_in = MinMaxScaler()
        self.mm_scaler_in.fit([self.min_in, self.max_in])

        self.mm_scaler_out = MinMaxScaler()
        self.mm_scaler_out.fit([self.min_out, self.max_out])

        # Create standardization scaler
        assert len(means) == len(stds) == 2
        self.mean_in, self.mean_out = means
        self.std_in, self.std_out = stds

        self.stand_scaler_in = StandardScaler()
        self.stand_scaler_in.fit([self.mean_in, self.mean_in])
        self.stand_scaler_in.scale_ = self.std_in

        self.stand_scaler_out = StandardScaler()
        self.stand_scaler_out.fit([self.mean_out * 0., self.mean_out * 0.]) # Ouput needs to be ZERO mean
        self.stand_scaler_out.scale_ = self.std_out

        self.__set_data_transform(self.transform_type)

    def get_data_attributes(self):
        """
        Geter function for the data attributes currently in use
        """
        if not self.transform_type:
            raise ValueError
        return ((self.mean_in, self.mean_out),
                (self.std_in, self.std_out),
                (self.max_in, self.max_out),
                (self.min_in, self.min_out))

    def __set_data_transform(self, transform_type):
        """
        Set one of the acceptable transform types
        minmax or standard
        """
        if not self.transform_type or not transform_type:
            raise ValueError("{} | {}".format(self.transform_type, transform_type))

        if transform_type == 'minmax':
            self.transform = (self.mm_scaler_in, self.mm_scaler_out)
        elif transform_type == 'standard':
            self.transform = (self.stand_scaler_in, self.stand_scaler_out)
        else:
            raise ValueError(self.transform_type)

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
            if self.transform_type:
                # TODO: Handle the series_len = 0 case
                x = self.transform[0].transform(x)
            X = np.append(X, np.expand_dims(np.array(x), axis=0), axis=0)
            Y = np.append(Y, np.expand_dims(np.array(y, ndmin=1), axis=0), axis=0)
        if self.transform_type:
            Y = self.transform[1].transform(Y)
        return X, Y


    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item

    def __get_data_idx(self, dl_id):
        if self.data_mode == ALL_VALID:
            return self.dataloaders[dl_id].get_valid_idx()
        elif self.data_mode == BOTH_SLIP:
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

    def __calculate_data_transforms(self):
        """
        Calculate the Normalization and Standardization tranforms for this data
        https://towardsai.net/p/data-science/how-when-and-why-should-you-normalize-standardize-rescale-your-data-3f083def38ff
        https://stats.stackexchange.com/questions/55999/is-it-possible-to-find-the-combined-standard-deviation
        """
        if not self.transform_type:
            raise ValueError

        # Get all mean and std of dataloaders
        mean_in = None
        mean_out = None
        max_in = None
        max_out = None
        min_in = None
        min_out = None
        num_dps = 0

        for dl in self.dataloaders:
            # Means
            m_in, m_out = dl.get_data_mean()
            mean_in = dl.size()*m_in if mean_in is None else mean_in + dl.size()*m_in
            mean_out = dl.size()*m_out if mean_out is None else mean_out + dl.size()*m_out

            # Maxs
            m_in, m_out = dl.get_data_max()
            max_in = m_in if max_in is None else np.maximum(max_in, m_in)
            max_out = m_out if max_out is None else np.maximum(max_out, m_out)

            # Mins
            m_in, m_out = dl.get_data_min()
            min_in = m_in if min_in is None else np.minimum(min_in, m_in)
            min_out = m_out if min_out is None else np.minimum(min_out, m_out)

            num_dps += dl.size()
        mean_in /= num_dps
        mean_out /= num_dps

        assert np.shape(mean_in) == np.shape(min_in) == np.shape(max_in)
        assert np.shape(mean_out) == np.shape(min_out) == np.shape(max_out)

        # Calculate the combined std of inputs and outputs
        std_in = None
        std_out = None
        for dl in self.dataloaders:
            # Stds
            m_in, m_out = dl.get_data_mean()
            s_in, s_out = dl.get_data_std()
            std_in = dl.size()*(s_in**2 + (m_in - mean_in)**2) if std_in is None \
                          else std_in + dl.size()*(s_in**2 + (m_in - mean_in)**2)
            std_out = dl.size()*(s_out**2 + (m_out - mean_out)**2) if std_out is None \
                          else std_out + dl.size()*(s_out**2 + (m_out - mean_out)**2)
        std_in = np.sqrt(std_in/num_dps)
        std_out = np.sqrt(std_out/num_dps)

        self.set_data_attributes(means=(mean_in, mean_out),
                                 stds=(std_in, std_out),
                                 maxs=(max_in, max_out),
                                 mins=(min_in, min_out))


if __name__ == "__main__":
    data_base = "/home/abhinavg/data/takktile/data-v1"
    dg = takktile_datagenerator(transform='standard')
    if dg.empty():
        print("The current Data generator is empty")
    dg.load_data_from_dir(dir_list=[data_base], series_len=20)
    print("Num Batches: {}".format(len(dg)))
    print("First Batch Comparison")
    for i in range(len(dg)):
        print(i)
        print(np.all(dg[i][0] == dg[i][0]))
        print(np.all(dg[i][1] == dg[i][1]))