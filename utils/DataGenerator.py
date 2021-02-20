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
from ConfigUtils import load_yaml
from DataAugment import takktile_data_augment
from utils import fft_real

# CONSTANTS
ALL_VALID = 1
BOTH_SLIP = 2
NO_SLIP = 3
SLIP_TRANS = 4
SLIP_ROT = 5
MAJOR_TRANS = 6
OBLIQUE_TRANS = 7


class takktile_datagenerator(tf.keras.utils.Sequence):
    """
    Keras datagenerator for takktile slip detection data

    Note: use load_data_from_dir function to populate data
    """

    def __init__(self, config, augment = None, balance=False, oversampling=False):
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

        eval_data: bool (legacy)
            indicated whether or not to create validation data from this dataset

        """
        self.batch_size = config['batch_size']
        self.dataloaders = []
        self.num_dl = 0
        self.shuffle = config['shuffle']
        self.data_mode = config['slip_filter']
        self.create_eval_data = config['eval_data']
        self.eval_len = 0
        self.transform_type = config['data_transform']['type'] if 'data_transform' in config else None
        self.series_len = config['series_len']
        self.config = config
        self.augment = augment
        self.balance_data = balance
        self.oversampling = oversampling

        if self.transform_type:
            assert self.transform_type == 'standard' or self.transform_type == 'minmax'

    ###########################################
    #  API FUNCTIONS
    ###########################################

    def empty(self):
        return self.num_dl <= 0

    def load_data_from_dir(self,
                           dir_list=[],
                           exclude=[]):
        """
        Recursive function to load data.mat files in directories with
        signature *takktile_* into a dataloader.

        Parameters
        ------------
        dir_list : list
            the root directory for the takktile data you wish to load
        exclude : list
            list of keywords that dataloaders paths should not contain
        """
        dir_list_ = dir_list[:]

        if len(dir_list) == 0:
            eprint("CANNOT load data generator with an empty list of directories: {}".format(dir_list))
            return

        for directory in dir_list_:
            if not os.path.isdir(directory):
                eprint("\t\t {}: {} is not a directory".format(self.load_data_from_dir.__name__, directory))
                return

        # Read Data from current directory
        while dir_list_:
            # Pop first directory name and create dataloader if its a valid folder
            current_dir = dir_list_.pop(0)
            valid_dir = True
            for name in exclude:
                if name in current_dir and valid_dir:
                    valid_dir = False
            data_file = current_dir + "/data.mat"
            if os.path.isfile(data_file) and "takktile_" in current_dir and valid_dir:
                self.dataloaders.append(takktile_dataloader(data_dir=current_dir,
                                                            config=self.config,
                                                            augment=self.augment))

            # Find all child directories of current directory and recursively load them
            data_dirs = [os.path.join(current_dir, o) for o in os.listdir(current_dir)
                    if os.path.isdir(os.path.join(current_dir, o))]
            for d in data_dirs:
                dir_list_.append(d)

        self.num_dl = len(self.dataloaders)
        if self.num_dl <= 0:
            raise ValueError("{}: Creating empty datagenrator".format(__file__))

        if self.transform_type:
            self.__calculate_data_transforms()

        # Create Eval Data
        if self.create_eval_data:
            self.eval_len = (self.__len__())//10
            self.create_eval_data = False

        # Calculate class number and ratios
        # Also calculate class diffs
        if not self.config['label_type'] == 'value':
            self.__class_nums = self.dataloaders[0].get_data_class_numbers(self.__get_data_idx(0))
            for i, dl in enumerate(self.dataloaders[1:]):
                self.__class_nums += dl.get_data_class_numbers(self.__get_data_idx(i+1))
            self.__class_ratios = self.__class_nums / float(np.mean(self.__class_nums))
            if self.oversampling:
                self.__class_diff = np.max(self.__class_nums) - self.__class_nums
            else:
                self.__class_diff = self.__class_nums - np.min(self.__class_nums)
            self.__class_diff = [d if n > 0 else 0 for n,d in zip(self.__class_nums, self.__class_diff)]

        # Reset and prepare data
        self.on_epoch_end()

    def reset_data(self):
        if self.empty():
            return

        self.dataloaders = []
        self.num_dl = 0
        self.__class_nums = None
        self.__class_ratios = None
        self.__class_diff = None

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

        if self.balance_data:
            self.__balance_generator_data()

    def evaluation_data(self):
        if not self.create_eval_data:
            eprint("No eval Data available")
        eval_len = self.eval_len
        self.eval_len = 0
        X, Y = self.__get_batches([self.__len__() -i-1 for i in range(eval_len)])
        self.eval_len = eval_len
        return X, Y

    def get_all_batches(self):
        a, b = self.__get_batches(range(self.__len__()))
        c = self.__get_vel_label_batches(range(self.__len__()))
        return a, b, c

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
        self.min_in = np.array(self.min_in)
        self.min_out = np.array(self.min_out)
        self.max_in = np.array(self.max_in)
        self.max_out = np.array(self.max_out)


        self.mm_scaler_in = MinMaxScaler()
        self.mm_scaler_in.fit([self.min_in, self.max_in])

        self.mm_scaler_out = MinMaxScaler()
        self.mm_scaler_out.fit([self.min_out, self.max_out])

        # Create standardization scaler
        assert len(means) == len(stds) == 2
        self.mean_in, self.mean_out = means
        self.std_in, self.std_out = stds
        self.mean_in = np.array(self.mean_in)
        self.mean_out = np.array(self.mean_out)
        self.std_in = np.array(self.std_in)
        self.std_out = np.array(self.std_out)

        self.stand_scaler_in = StandardScaler()
        self.stand_scaler_in.fit([self.mean_in, self.mean_in])
        self.stand_scaler_in.scale_ = self.std_in

        self.stand_scaler_out = StandardScaler()
        if self.config['data_transform']['output_mean_zero']:
            self.stand_scaler_out.fit([self.mean_out * 0., self.mean_out * 0.]) # Ouput mean should be zero
        else:
            self.stand_scaler_out.fit([self.mean_out, self.mean_out])
        self.stand_scaler_out.scale_ = self.std_out

        self.__set_data_transform(self.transform_type)

    def load_data_attributes_from_config(self):
        mean = self.config['data_transform']['mean']
        std = self.config['data_transform']['std']
        max_ = self.config['data_transform']['max']
        min_ = self.config['data_transform']['min']
        self.set_data_attributes(mean, std, max_, min_)

    def load_data_attributes_to_config(self):
        mean, std, max_, min_ = self.get_data_attributes()
        self.config['data_transform']['mean'] = (mean[0].tolist(), mean[1].tolist())
        self.config['data_transform']['std'] = (std[0].tolist(), std[1].tolist())
        self.config['data_transform']['max'] = (max_[0].tolist(), max_[1].tolist())
        self.config['data_transform']['min'] = (min_[0].tolist(), min_[1].tolist())

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

    def get_inverse_transform(self, inputs=[], outputs=[]):
        x = np.copy(inputs); y = []
        if 'data_format' in self.config and self.config['data_format'] == 'vector3D':
            x = np.reshape(x, (-1, self.series_len, 6))
        if self.transform_type:
            if len(inputs) > 0 and 'data_format' in self.config and self.config['data_format'] != 'freq_image':
                for i, inp in enumerate(x):
                    x[i] = self.transform[0].inverse_transform(inp)
            if len(outputs) > 0:
                y = self.transform[1].inverse_transform(outputs)
        return x, y

    def get_class_nums(self):
        if not self.config['label_type'] == 'value':
            ret = self.__class_nums[:]
            if self.balance_data:
                if self.oversampling: ret += self.__class_diff
                else: ret-= self.__class_diff
            return self.__class_nums
        else:
            eprint("Cannot get class nums during regression tasks")
            raise NotImplementedError


    ###########################################
    #  PRIVATE FUNCTIONS
    ###########################################


    def __len__(self):
        if self.empty():
            return 0
        num = 0
        for data_idx in self.dl_data_idx:
            num += len(data_idx)
        real_len = int(num / self.batch_size)
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
            if self.transform_type: # X is transformed before concatenation
                # TODO: Handle the series_len = 0 case
                x = self.transform[0].transform(x)
            X = np.append(X, np.expand_dims(np.array(x), axis=0), axis=0)
            Y = np.append(Y, np.expand_dims(np.array(y, ndmin=1), axis=0), axis=0)
        if self.transform_type:
            Y = self.transform[1].transform(Y)
        # Convert to images for 3D convolution
        if 'data_format' in self.config and self.config['data_format'] == 'vector3D':
            X = np.reshape(X, (-1,self.series_len,2,3,1))
            X = np.flip(X, 2)
            X[:,:,0,:,:] = np.flip(X[:,:,0,:,:], 2)
        # Convert to Frequency domain images with fft algorithm
        elif 'data_format' in self.config and self.config['data_format'] == 'freq_image':
            X = np.reshape(X, (-1,2,3,self.series_len))
            fft_len = (np.shape(X)[-1]/2) + 1
            X = fft_real(X, axis=-1)[:,:,:,:fft_len]
            X = np.flip(X, 1)
            X[:,0,:,:] = np.flip(X[:,0,:,:], 1)
        return X, Y

    def __get_vel_label_batches(self, batches=[]):
        VEL_ = np.empty([0])
        for i in batches:
            VEL = self.__get_vel_label_batch(i)
            VEL_ = VEL if VEL_.size == 0 else np.append(VEL_, VEL, axis=0)
        if VEL_.size == 0:
            return
        return VEL_

    def __get_vel_label_batch(self, batch_index):
        if self.empty() or batch_index >= self.__len__() or batch_index < 0:
            eprint("Index out of bounds")
            raise ValueError("Index {} out of bounds for length {}".format(batch_index, self.__len__()))

        # Fetching data from [index*bs] -> [(index+1)*bs]
        indices = range(batch_index*self.batch_size, (batch_index+1)*self.batch_size)
        vel_ = self.dataloaders[0].get_velocity_label(0)
        VEL = np.empty([0, np.size(vel_)])
        for i in indices:
            vel = None
            dl_id = 0
            for dl_id in self.dl_idx:
                # Find the bin that i belongs to
                if i >= len(self.dl_data_idx[dl_id]):
                    i -= len(self.dl_data_idx[dl_id])
                else:
                    break
            vel = self.dataloaders[dl_id].get_velocity_label(self.dl_data_idx[dl_id][i])
            VEL = np.append(VEL, np.expand_dims(np.array(vel, ndmin=1), axis=0), axis=0)
        return VEL

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
        elif self.data_mode == MAJOR_TRANS:
            return self.dataloaders[dl_id].get_major_slip_idx()
        elif self.data_mode == OBLIQUE_TRANS:
            return self.dataloaders[dl_id].get_oblique_slip_idx()
        else:
            eprint("Unrecognised data mode: {}".format(self.data_mode))
            raise ValueError("Unrecognised data mode")

    def __balance_generator_data(self):
        if self.config['label_type'] == 'value':
            eprint("Data Balancing not implemented for regression")
            return

        diffs = self.__class_diff[:]
        for i, curr_diff in enumerate(diffs):
            curr_label = np.zeros(len(diffs))
            curr_label[i] = 1
            # print("Filling Label: {}".format(curr_label))
            for dp in range(curr_diff):
                found = False
                while not found:
                    rand_dl_idx = np.random.choice(self.dl_idx)
                    rand_dl_data_idx = self.dataloaders[rand_dl_idx].get_random_data_idx_with_label(curr_label)
                    if not rand_dl_data_idx == None:
                        # Add random label data to dataloader
                        if self.oversampling:
                            self.dl_data_idx[rand_dl_idx].append(rand_dl_data_idx)
                            if self.shuffle: # Shuffle dataloader data list
                                np.random.shuffle(self.dl_data_idx[rand_dl_idx])
                            found = True
                        elif rand_dl_data_idx in self.dl_data_idx[rand_dl_idx]:
                            self.dl_data_idx[rand_dl_idx].remove(rand_dl_data_idx)
                            found = True
                        # print("Found: {} in {}".format(rand_dl_data_idx, rand_dl_idx))


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
    config = load_yaml("./configs/base_config_tcn.yaml")
    config = config['data']
    dg = takktile_datagenerator(config, takktile_data_augment(config), balance=False)
    if dg.empty():
        print("The current Data generator is empty")
    # Load data into datagen
    dir_list = [config['data_home'] + config['train_dir']]
    dg.load_data_from_dir(dir_list=dir_list, exclude=config['train_data_exclude'])

    cn = dg.get_class_nums()
    print("Data distribution: {}".format([cn[0], np.sum(cn[1:])]))
    print([cn[1] + cn[3] + cn[5] + cn[7], cn[2] + cn[4] + cn[6] + cn[8] ])


