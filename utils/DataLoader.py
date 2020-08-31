#!/usr/bin/env python2.7

"""
DataLoader.py

Util file for handling takktile recorded data

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
import copy

# GLOBAL VARIABLES
# SPEED_THRESH = 2.5                # Relatively tuned (High is strict for slip)
SLIP_STD_VALID_MULTIPLIER = 0.625  # Tuned to lie between 0.62 and 0.68 (High is strict)
TEMPORAL_DIR_STD = math.pi*3/18   # Error upto 30 degree is allowed (High is lenient)
VICON_MODE = "vicon"
IMU_MODE = "imu"
FLOW_MODE = "flow"

# Normalization CONSTANTS
SPEED_SCALE = 1.0
ANG_SPEED_SCALE = 20.0

class takktile_dataloader(object):
    """
        Takktile data loading class (.mat files)

        This file processes the available tactile sensor data
        to follow certain conditions:
        1.  Create an iterable list of indices with valid data-series of length
            series_len by reading the dictionary data in data_dir
        2.  Slip speed above __speed_thresh is considered slipping, class creates a list
            of slip and non-slip indices. Each of the datapoints in the time-series
            data must be valid datapoint.
        3.  Valid data: any datapoint where most flow vectors are in agreement
    """

    def __init__(self, data_dir, input_len = 20, create_hist=False, mat_format = True):
        self.series_len = input_len

        # Load Data
        self.__load_data_dir(data_dir, mat_format)

        # Set states
        self.create_hist = create_hist and self.__get_mode() == FLOW_MODE

        # Set Global Variables
        self.__speed_thresh = 2.5 if self.__get_mode() == FLOW_MODE else 0.1

        if self.empty():
            eprint("\t\t Not enough data in directory: {}".format(data_dir))
            return

        # Process Data
        self.__process_data()

        # Create histogram
        if self.create_hist:
            # All indices
            self.__create_slip_hist(indices=self.get_all_idx())
            self.save_slip_hist(dir + "/slip_hist_all.png")
            # Valid indices
            self.__create_slip_hist(indices=self.get_valid_idx())
            self.save_slip_hist(dir + "/slip_hist_valid.png")
            # Slip stream indices
            self.__create_slip_hist(indices=self.get_slip_stream_idx())
            self.save_slip_hist(dir + "/slip_hist_slip.png")
            # No Slip stream indices
            self.__create_slip_hist(indices=self.get_no_slip_stream_idx())
            self.save_slip_hist(dir + "/slip_hist_no_slip.png")


    ###########################################
    #  API FUNCTIONS
    ###########################################
    def save_slip_hist(self, location):
        """
        Save the 2D histogram of the slip data at location
            location: address including the name of the saved image
        """
        if self.empty():
            eprint("\t\t Cannot save histogram: empty loader")
            return
        if not self.create_hist:
            eprint("\t\t Cannot save histogram: histogram disabled")
            return
        self.slip_hist.savefig(location, dpi=self.slip_hist.dpi)

    def size(self):
        if self.empty():
            return 0
        return int(self.__data['num'])

    def empty(self):
        return self.__data['num'] <= self.series_len

    def get_mode(self):
        return self.__get_mode()

    def get_all_idx(self):
        """
        Return all data indices
        """
        if self.empty():
            return []
        return range(self.size())

    def get_valid_idx(self):
        """
        Return valid data indices
        """
        if self.empty():
            return []
        return copy.copy(self.valid_idx)

    def get_slip_idx(self):
        """
        Return translation slip only data indices
        """
        if self.empty():
            return []
        return copy.copy(self.slip_idx)

    def get_rot_idx(self):
        """
        Return rotation slip only data indices
        """
        if self.empty():
            return []
        return copy.copy(self.rot_idx)

    def get_slip_n_rot_idx(self):
        """
        Return coupled slip data indices
        """
        if self.empty():
            return []
        return copy.copy(self.coupled_slip_idx)

    def get_no_slip_idx(self):
        """
        Return no slip data indices
        """
        if self.empty():
            return []
        return copy.copy(self.no_slip_idx)

    def get_slip_stream_idx(self):
        """
        Return slip only data indices
        """
        if self.empty():
            return []
        return copy.copy(self.slip_stream_idx)

    def get_no_slip_stream_idx(self):
        """
        Return no slip data indices
        """
        if self.empty():
            return []
        return copy.copy(self.no_slip_stream_idx)

    ###########################################
    #  PRIVATE FUNCTIONS
    ###########################################

    def __getitem__(self, idx):
        """
        Return learning data at index
            Data Format: [pressure], ([slip_dir], [slip_speed])
        """
        if self.empty() or not isinstance(idx, (int, long)):
            eprint("\t\t Incorrect index access: {}".format(idx))
            return ()

        if idx<0 or idx>=self.size():
            eprint("\t\t Incorrect index access: {}".format(idx))
            return ()

        ret_list = range(idx-self.series_len+1, idx+1)
        if self.__get_mode() == FLOW_MODE:
            return self.__get_pressure(ret_list), \
                   (self.__get_slip_angle(idx), \
                    self.__get_slip_speed(idx))
        else:
            return self.__get_pressure(ret_list), \
                   (self.__get_slip_dir(idx)[0], \
                    self.__get_slip_dir(idx)[1], \
                    self.__get_ang_vel(idx))

    def __len__(self):
        return self.size()

    def __load_data_dir(self, data_dir, mat_format):
        print("Loading data file {}".format(data_dir))
        self.__data_dir = data_dir
        self.mat_format = mat_format
        self.__data = sio.loadmat(data_dir + "/data.mat", appendmat=mat_format)
        # Check if the data contains all fields
        if 'num' not in self.__data or \
           'mode' not in self.__data or \
           'material' not in self.__data or \
           'temp' not in self.__data or \
           'pressure' not in self.__data or \
           'time' not in self.__data or \
           'slip' not in self.__data:
            eprint("\t\t {} has outdated data, skipping.....".format(data_dir))
            self.__data['num'] = 0
            return
        # Generate angle data
        slip_dir_array = self.__get_slip_dir(self.get_all_idx())
        self.__slip_angle_data = np.array([[math.atan2(a[1], a[0])] for a in slip_dir_array])

        # calculate time period of data recording
        if self.size() >= 2:
            self.__data_period = self.__get_time(1) - self.__get_time(0)

    def __create_slip_hist(self, show=False, indices=[]):
        if (np.array(indices)<0).any() or \
           (np.array(indices) >= self.size()).any():
            eprint("\t\t create histogram: Incorrect indices")
            self.slip_hist = None
            return

        hist_data = self.__get_slip_dir(indices) * \
                    self.__get_slip_speed(indices)
        self.slip_hist = plt.figure(figsize=(10,10))
        plt.hist2d(hist_data[:, 0], hist_data[:, 1], bins=100)
        t = np.linspace(0,np.pi*2,100)
        plt.plot(self.__speed_thresh*np.cos(t), self.__speed_thresh*np.sin(t), linewidth=1)
        plt.title("Slip Histogram of {} points in {}\
                  ".format(len(indices), self.__data_dir))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        if show:
            self.__show_slip_hist()

    def __show_slip_hist(self):
        if not self.slip_hist:
            self.__create_slip_hist(show=True)
        else:
            plt.show(self.slip_hist)

    def __process_data(self):
        # define index lists
        self.valid_idx = []
        self.no_slip_idx = []        # No slip on current idx
        self.slip_idx = []           # translation Slip only on current idx
        self.rot_idx = []            # Rotational slip only on current idx
        self.coupled_slip_idx = []   # Rotational and translational slip combined
        self.no_slip_stream_idx = [] #  No slip in data[idx-len+1]->data[idx]
        self.slip_stream_idx = []    #  Slip in data[idx-len+1]->data[idx]

        valid_counter = 0 # Counts the trail of valid idx
        slip_counter = 0  # Counts the trail of trans slip only idx
        rot_counter = 0   # counts the trail of rot slip only idx
        no_slip_counter = 0 # counts the trail of coupled slip idx
        # Iterate over each datapoint
        for idx in range(self.size()):
            if self.__is_valid(idx):
                valid_counter += 1
                if self.__get_slip_speed(idx) < self.__speed_thresh and \
                   abs(self.__get_ang_vel(idx)) < self.__speed_thresh:
                    no_slip_counter += 1
                    slip_counter = 0
                elif self.__get_slip_speed(idx) > self.__speed_thresh and \
                   abs(self.__get_ang_vel(idx)) < self.__speed_thresh:
                    no_slip_counter = 0
                    slip_counter += 1
                    rot_counter = 0
                elif self.__get_slip_speed(idx) > self.__speed_thresh and \
                   abs(self.__get_ang_vel(idx)) > self.__speed_thresh:
                    no_slip_counter = 0
                    slip_counter = 0
                    rot_counter += 1
                else:
                    no_slip_counter = 0
                    slip_counter += 1
                    rot_counter += 1
            else:
                valid_counter = 0
                no_slip_counter = 0
                slip_counter = 0

            # data[i-s] to data[i] is the valid data for index i
            # Current data is only valid if the data series has all valid indices
            if valid_counter >= self.series_len:
                # Storing every valid index
                self.valid_idx.append(idx)
                # Storing slip and non slip index
                if no_slip_counter > 0:
                    self.no_slip_idx.append(idx)
                elif slip_counter > 0 and rot_counter == 0:
                    self.slip_idx.append(idx)
                elif slip_counter == 0 and rot_counter > 0:
                    self.rot_idx.append(idx)
                else:
                    self.coupled_slip_idx.append(idx)
                # Store slip and non-slip stream indices
                if no_slip_counter >= self.series_len:
                    self.no_slip_stream_idx.append(idx)
                if slip_counter >= self.series_len or rot_counter >= self.series_len:
                    # temporal_std_dir = np.std(self.__get_slip_angle(range(idx+1-self.series_len, idx+1)))
                    # if temporal_std_dir < TEMPORAL_DIR_STD:
                    self.slip_stream_idx.append(idx)

    def __is_valid(self, idx):
        """
        Check for the current time step whether the flow vectors are in agreement
        or not, ie, is the current measurement valid or not

        Parameters
        ----------
        idx : int
            The index for the data to check the validity of

        Returns
        ----------
        valid : bool
            Is the data at this index valid?
        """
        if idx<0 or idx>=self.size():
            return False

        if self.__get_mode() != FLOW_MODE:
            return True

        slip_speed = self.__get_slip_speed(idx)
        slip_vector = self.__get_slip_dir(idx) * slip_speed
        slip_std = self.__get_slip_std(idx)

        max_dim = [i for i in range(len(slip_vector)) if slip_vector[i]>self.__speed_thresh]

        if (slip_std*SLIP_STD_VALID_MULTIPLIER > slip_vector)[max_dim].any():
            return False

        return True

    def __get_pressure(self, idx):
        return self.__data['pressure'][idx]

    def __get_mode(self):
        return self.__data['mode']

    def __get_temp(self, idx):
        return self.__data['temp'][idx]

    def __get_time(self, idx):
        return self.__data['time'][idx]

    def __get_slip_dir(self, idx):
        if self.__get_mode() == FLOW_MODE:
            return self.__data['slip'][idx][1:3]
        else:
            return self.__data['slip'][idx][0:2]

    def __get_slip_angle(self, idx):
        return self.__slip_angle_data[idx]/SPEED_SCALE

    def __get_slip_speed(self, idx):
        if self.__get_mode() == FLOW_MODE:
            return self.__data['slip'][idx][0]
        else:
            if isinstance(idx, (int, long)):
                return np.linalg.norm(self.__data['slip'][idx,0:2])
            return np.linalg.norm(self.__data['slip'][idx,0:2], axis=1)

    def __get_ang_vel(self, idx):
        if self.__get_mode() != FLOW_MODE:
            return self.__data['slip'][idx][2] / ANG_SPEED_SCALE
        else:
            raise self.__data['slip'][idx][2] * 0   #ZEROS

    def __get_slip_std(self, idx):
        if self.__get_mode() == FLOW_MODE:
            return self.__data['slip'][idx][3:5]
        else:
            raise NotImplementedError


if __name__ == "__main__":
    data_base = "/home/abhinavg/data/takktile/"
    data_dirs = [os.path.join(data_base, o) for o in os.listdir(data_base)
                if os.path.isdir(os.path.join(data_base,o)) and "takktile_" in o ]
    dataloaders = []
    for dir in data_dirs:
        dataloaders.append(takktile_dataloader(dir))
        if not dataloaders[-1].empty():
            # print("Number of datapoints that have a seq num sized trail of slip data")
            print([dataloaders[-1][id][1] for id in dataloaders[-1].get_slip_idx()])