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

from ConfigUtils import load_yaml

# GLOBAL CONSTANTS
SPEED_THRESH_FLOW = 2.5                # Relatively tuned (High is strict for slip)
SPEED_THRESH_ = 0.1
SLIP_STD_VALID_MULTIPLIER = 0.625  # Tuned to lie between 0.62 and 0.68 (High is strict)
TEMPORAL_DIR_STD = math.pi*3/18   # Error upto 30 degree is allowed (High is lenient)
VICON_MODE = "vicon"
IMU_MODE = "imu"
FLOW_MODE = "flow"

# Transformation CONSTANTS (Legacy)
SPEED_SCALE = 1.0
ANG_SPEED_SCALE = 1.0
PRESSURE_SCALE = 1.0
PRESSURE_OFFSET = 0

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

    def __init__(self, data_dir,
                       config):
        """
        Parameters
        --------------
        data_dir: str
            must be a valid takktile data directory with data.mat file present
        input_len: int
            The temporal length of the time series data used as input for learning
        create_hist: bool
            indicated whether histogram should be generated or not
        rotation: bool
            indicated whether rotation speed should be included in the output or not
        translation: bool
            indicated whether translation speed should be included in the output or not
        """
        self.series_len = config['series_len']
        self.config = config.copy()

        # Load Data
        self.__load_data_dir(data_dir, mat_format=(config['format']=='mat'))

        # Set states
        self.create_hist = self.config['histogram']['create']
        self.get_translation = False
        self.get_rotation = False

        # Setparameters
        self.__speed_thresh = config['slip_thresh']['flow'] if self.__get_mode() == FLOW_MODE \
                                                else config['slip_thresh']['speed']*SPEED_SCALE

        if self.empty():
            eprint("\t\t Not enough data in directory: {}".format(data_dir))
            return

        # Process Data
        self.__process_data()

        # Create histogram
        if self.create_hist == True:
            # Which data to use?
            data_mode = self.config['histogram']['slip_filter']
            if data_mode == 1:
                indices = self.get_valid_idx()
            elif data_mode == 2:
                indices = self.get_slip_n_rot_idx()
            elif data_mode == 3:
                indices = self.get_no_slip_idx()
            elif data_mode == 4:
                indices = self.get_slip_idx()
            elif data_mode == 5:
                indices = self.get_rot_idx()
            else:
                eprint("Unrecognised data mode: {}".format(data_mode))
                raise ValueError("Unrecognised data mode")
            # Create and save
            self.__create_slip_hist(indices=indices)
            if self.config['histogram']['save'] == True:
                self.save_slip_hist(data_dir)

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
                   np.array([self.__get_slip_angle(idx), \
                             self.__get_slip_speed(idx)])
        else:
            return self.__get_pressure(ret_list), \
                    self.__get_label(idx=idx)

    def get_velocity_label(self, idx):
        """
        Return velocity data at index
            Data Format: slip_x, slip_y, rot
        """
        if self.empty() or not isinstance(idx, (int, long)):
            eprint("\t\t Incorrect index access: {}".format(idx))
            return ()

        if idx<0 or idx>=self.size():
            eprint("\t\t Incorrect index access: {}".format(idx))
            return ()

        if self.__get_mode() == FLOW_MODE:
            raise ValueError("Cannot get velocity in FLOW_MODE")
        else:
            return np.array([self.__get_slip_dir(idx)[0], \
                            self.__get_slip_dir(idx)[1], \
                            self.__get_ang_vel(idx)])

    ###########################################
    #  API FUNCTIONS
    ###########################################
    def save_slip_hist(self, directory):
        """
        Save the 2D histogram of the slip data at directory
            directory: address including the name of the saved image
        """
        if self.empty():
            eprint("\t\t Cannot save histogram: empty loader")
            return
        if not self.create_hist:
            eprint("\t\t Cannot save histogram: histogram disabled")
            return
        if self.slip_vel_hist:
            self.slip_vel_hist.savefig(directory + "/vel_hist.png", dpi=self.slip_vel_hist.dpi)
        if self.slip_ang_vel_hist:
            self.slip_ang_vel_hist.savefig(directory + "/ang_vel_hist.png", dpi=self.slip_ang_vel_hist.dpi)

    def size(self):
        if self.empty():
            return 0
        return int(self.__data['num'])

    def empty(self):
        return self.__data['num'] <= self.series_len

    def get_data_class_numbers(self):
        """
        Get means of inputs and outputs
        return: (input_means, output_means)
        """
        if self.config['label_type'] == 'value':
            return []

        all_indices = self.get_valid_idx()

        # Collect the labels
        labels = self.__get_label(all_indices[0])
        for i in all_indices[1:]:
            labels = np.vstack((labels, self.__get_label(i)))

        return np.sum(labels, axis=0)

    def get_data_mean(self):
        """
        Get means of inputs and outputs
        return: (input_means, output_means)
        """
        all_indices = self.get_all_idx()

        if self.__get_mode() == FLOW_MODE:
            return np.mean(self.__get_pressure(all_indices), axis=0), \
                   np.array([np.mean(self.__get_slip_angle(all_indices)), \
                             np.mean(self.__get_slip_speed(all_indices))])
        else:
            # Collect the labels
            if self.config['label_type'] == 'value':
                labels = self.__get_label(all_indices[0])
                for i in all_indices[1:]:
                    labels = np.vstack((labels, self.__get_label(i)))

                return np.mean(self.__get_pressure(all_indices), axis=0), \
                       np.mean(labels, axis=0)
            else:
                return np.mean(self.__get_pressure(all_indices), axis=0), \
                       np.zeros_like(self.__get_label(all_indices[0]))


    def get_data_std(self):
        """
        Get standard deviations of inputs and outputs
        return: (input_means, output_means)
        """
        all_indices = self.get_all_idx()
        if self.__get_mode() == FLOW_MODE:
            return np.std(self.__get_pressure(all_indices), axis=0), \
                    np.array([np.std(self.__get_slip_angle(all_indices)), \
                              np.std(self.__get_slip_speed(all_indices))])
        else:
            # Collect the labels
            if self.config['label_type'] == 'value':
                labels = self.__get_label(all_indices[0])
                for i in all_indices[1:]:
                    labels = np.vstack((labels, self.__get_label(i)))

                return np.std(self.__get_pressure(all_indices), axis=0), \
                       np.std(labels, axis=0)
            else:
                return np.std(self.__get_pressure(all_indices), axis=0), \
                       np.ones_like(self.__get_label(all_indices[0]))

    def get_data_min(self):
        """
        Get standard deviations of inputs and outputs
        return: (input_means, output_means)
        """
        all_indices = self.get_all_idx()
        if self.__get_mode() == FLOW_MODE:
            return np.min(self.__get_pressure(all_indices), axis=0), \
                    np.array([np.min(self.__get_slip_angle(all_indices)), \
                              np.min(self.__get_slip_speed(all_indices))])
        else:
            # Collect the labels
            if self.config['label_type'] == 'value':
                labels = self.__get_label(all_indices[0])
                for i in all_indices[1:]:
                    labels = np.vstack((labels, self.__get_label(i)))

                return np.min(self.__get_pressure(all_indices), axis=0), \
                       np.min(labels, axis=0)
            else:
                return np.min(self.__get_pressure(all_indices), axis=0), \
                       np.zeros_like(self.__get_label(all_indices[0]))

    def get_data_max(self):
        """
        Get standard deviations of inputs and outputs
        return: (input_means, output_means)
        """
        all_indices = self.get_all_idx()
        if self.__get_mode() == FLOW_MODE:
            return np.max(self.__get_pressure(all_indices), axis=0), \
                    np.array([np.max(self.__get_slip_angle(all_indices)), \
                              np.max(self.__get_slip_speed(all_indices))])
        else:
            # Collect the labels
            if self.config['label_type'] == 'value':
                labels = self.__get_label(all_indices[0])
                for i in all_indices[1:]:
                    labels = np.vstack((labels, self.__get_label(i)))

                return np.max(self.__get_pressure(all_indices), axis=0), \
                       np.max(labels, axis=0)
            else:
                return np.max(self.__get_pressure(all_indices), axis=0), \
                       np.ones_like(self.__get_label(all_indices[0]))

    def plot_pressure(self):
        all_indices = self.get_all_idx()
        press = self.__get_pressure(all_indices)
        time = self.__get_time(all_indices)
        self.press_plot = plt.figure(figsize=(10,10))
        plt.plot(time[:500], press[:500,:])
        plt.xlabel('time')
        plt.ylabel('pressure')
        plt.show()


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
            self.slip_vel_hist = None
            return

        if not indices:
            indices = self.get_all_idx()

        if self.__get_mode() == FLOW_MODE:
            hist_data = [self.__get_slip_dir(index)*self.__get_slip_speed(index) \
                         for index in indices]
            self.slip_vel_hist = plt.figure(figsize=(10,10))
            plt.hist2d([d[0] for d in hist_data] , [d[1] for d in hist_data], bins=50)
            t = np.linspace(0,np.pi*2,100)
            plt.plot(self.__speed_thresh*np.cos(t), self.__speed_thresh*np.sin(t), linewidth=1)
            plt.title("Slip Histogram of {} points in {}\
                    ".format(len(indices), self.__data_dir))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.axis('equal')
        else:
            # Slip Speed Hist
            hist_data = [self.__get_slip_dir(index) / SPEED_SCALE \
                         for index in indices]

            self.slip_vel_hist = plt.figure(figsize=(10,10))
            plt.hist2d([d[0] for d in hist_data] , [d[1] for d in hist_data], bins=50)
            t = np.linspace(0,np.pi*2,100)
            plt.plot(self.__speed_thresh*np.cos(t), self.__speed_thresh*np.sin(t), linewidth=1)
            num_trans_slip = len(self.get_slip_idx() + self.get_slip_n_rot_idx())
            plt.title("Slip Histogram of {} indices with {} slip and {} non-slip points\
                    ".format(len(indices), num_trans_slip, len(indices)-num_trans_slip))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.axis('equal')

            # Angular Velocity Histogram
            hist_ang_data = [self.__get_ang_vel(index) / ANG_SPEED_SCALE \
                         for index in indices]
            self.slip_ang_vel_hist = plt.figure(figsize=(10,10))
            plt.hist(hist_ang_data, bins=100)
            num_rot_slip = len(self.get_rot_idx() + self.get_slip_n_rot_idx())
            plt.title("Slip Angular Velocity Histogram of {} indices with {} slip and {} non-slip points\
                    ".format(len(indices), num_rot_slip, len(indices)-num_rot_slip))
            plt.xlabel('Angular Vel rad/s')
            plt.ylabel('Frequency')

        if show:
            self.__show_slip_hist()

    def __show_slip_hist(self):
        if not self.slip_vel_hist:
            self.__create_slip_hist(show=True)
        else:
            plt.show(self.slip_vel_hist)
        if not self.slip_ang_vel_hist:
            self.__create_slip_hist(show=True)
        else:
            plt.show(self.slip_ang_vel_hist)

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
                   abs(self.__get_ang_vel(idx)) < self.config['slip_thresh']['angular_speed']:
                    no_slip_counter += 1
                    slip_counter = 0
                    rot_counter = 0
                elif self.__get_slip_speed(idx) > self.__speed_thresh and \
                   abs(self.__get_ang_vel(idx)) < self.config['slip_thresh']['angular_speed']:
                    no_slip_counter = 0
                    slip_counter += 1
                    rot_counter = 0
                elif self.__get_slip_speed(idx) < self.__speed_thresh and \
                   abs(self.__get_ang_vel(idx)) > self.config['slip_thresh']['angular_speed']:
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

    def booleans_to_categorical(self, b_list):
        """
            Input is a list of booleans and the output is a categorical array with
            a '1' in the corresponding position of the array
        """
        _size, base = np.shape(b_list)

        ret = np.zeros([base**_size])

        b_list = [[int(i) for i in j] for j in b_list]
        if not np.sum(b_list) == _size: # Weed out invalid boolean lists
            raise ValueError("Invalid boolean list {}".format(b_list))

        class_num = 0
        b_list_int = [np.argmax(k) for k in b_list]
        for l in range(len(b_list_int)):
            class_num += b_list_int[l] * (base**l)
        ret[class_num] = 1
        return ret

    def __get_label(self, idx):
        """
            Return the slip label values based on label type and dimension
        """
        if self.__get_mode() == FLOW_MODE:
            raise ValueError("Cannot use get label in flow mode")

        if self.config['label_type'] == 'value':
            if self.config['label_dimension'] == 'all':
                return np.array([self.__get_slip_dir(idx)[0], \
                                self.__get_slip_dir(idx)[1], \
                                self.__get_ang_vel(idx)])
            elif self.config['label_dimension'] == 'translation':
                return np.array([self.__get_slip_dir(idx)[0], \
                                self.__get_slip_dir(idx)[1]])
            elif self.config['label_dimension'] == 'rotation':
                return np.array([self.__get_ang_vel(idx)])
            elif self.config['label_dimension'] == 'x':
                return np.array(self.__get_slip_dir(idx)[0])
            elif self.config['label_dimension'] == 'y':
                return np.array(self.__get_slip_dir(idx)[1])
            else:
                raise ValueError("Invalid label dimension {}".format(self.config['label_dimension']))
        # Direction only
        elif self.config['label_type'] == 'direction':
            if self.config['label_dimension'] == 'all':
                return self.booleans_to_categorical(
                                [[self.__get_slip_dir(idx)[0] > self.__speed_thresh,
                                not self.__get_slip_dir(idx)[0] > self.__speed_thresh and \
                                not self.__get_slip_dir(idx)[0] < -self.__speed_thresh,
                                self.__get_slip_dir(idx)[0] < -self.__speed_thresh],
                                [self.__get_slip_dir(idx)[1] > self.__speed_thresh,
                                not self.__get_slip_dir(idx)[1] > self.__speed_thresh and \
                                not self.__get_slip_dir(idx)[1] < -self.__speed_thresh,
                                self.__get_slip_dir(idx)[1] < -self.__speed_thresh],
                                [self.__get_ang_vel(idx) > self.config['slip_thresh']['angular_speed'],
                                not self.__get_ang_vel(idx) > self.config['slip_thresh']['angular_speed'] and \
                                not self.__get_ang_vel(idx) < -self.config['slip_thresh']['angular_speed'],
                                self.__get_ang_vel(idx) < -self.config['slip_thresh']['angular_speed']]])
            elif self.config['label_dimension'] == 'translation':
                return self.booleans_to_categorical(
                                [[self.__get_slip_dir(idx)[0] > self.__speed_thresh,
                                not self.__get_slip_dir(idx)[0] > self.__speed_thresh and \
                                not self.__get_slip_dir(idx)[0] < -self.__speed_thresh,
                                self.__get_slip_dir(idx)[0] < -self.__speed_thresh],
                                [self.__get_slip_dir(idx)[1] > self.__speed_thresh,
                                not self.__get_slip_dir(idx)[1] > self.__speed_thresh and \
                                not self.__get_slip_dir(idx)[1] < -self.__speed_thresh,
                                self.__get_slip_dir(idx)[1] < -self.__speed_thresh]])
            elif self.config['label_dimension'] == 'rotation':
                return self.booleans_to_categorical(
                                [self.__get_ang_vel(idx) > self.config['slip_thresh']['angular_speed'],
                                not self.__get_ang_vel(idx) > self.config['slip_thresh']['angular_speed'] and \
                                not self.__get_ang_vel(idx) < -self.config['slip_thresh']['angular_speed'],
                                self.__get_ang_vel(idx) < -self.config['slip_thresh']['angular_speed']])
            elif self.config['label_dimension'] == 'x':
                return self.booleans_to_categorical(
                                [[self.__get_slip_dir(idx)[0] > self.__speed_thresh,
                                not self.__get_slip_dir(idx)[0] > self.__speed_thresh and \
                                not self.__get_slip_dir(idx)[0] < -self.__speed_thresh,
                                self.__get_slip_dir(idx)[0] < -self.__speed_thresh]])
            elif self.config['label_dimension'] == 'y':
                return self.booleans_to_categorical(
                                [[self.__get_slip_dir(idx)[1] > self.__speed_thresh,
                                not self.__get_slip_dir(idx)[1] > self.__speed_thresh and \
                                not self.__get_slip_dir(idx)[1] < -self.__speed_thresh],
                                self.__get_slip_dir(idx)[1] < -self.__speed_thresh])
            else:
                raise ValueError("Invalid label dimension {}".format(self.config['label_dimension']))
        # Slip only
        elif self.config['label_type'] == 'slip':
            if self.config['label_dimension'] == 'all':
                slip = abs(self.__get_slip_dir(idx)[0]) > self.config['slip_thresh']['speed'] or \
                        abs(self.__get_slip_dir(idx)[1]) > self.config['slip_thresh']['speed'] or \
                        abs(self.__get_ang_vel(idx)) > self.config['slip_thresh']['angular_speed']
                return self.booleans_to_categorical(
                                [[not slip, slip]])
            elif self.config['label_dimension'] == 'translation':
                slip = abs(self.__get_slip_dir(idx)[0]) > self.config['slip_thresh']['speed'] or \
                        abs(self.__get_slip_dir(idx)[1]) > self.config['slip_thresh']['speed']
                return self.booleans_to_categorical(
                                [[not slip, slip]])
            elif self.config['label_dimension'] == 'rotation':
                slip = abs(self.__get_ang_vel(idx)) > self.config['slip_thresh']['angular_speed']
                return self.booleans_to_categorical(
                                [[not slip, slip]])
            elif self.config['label_dimension'] == 'x':
                slip = abs(self.__get_slip_dir(idx)[0]) > self.config['slip_thresh']['speed']
                return self.booleans_to_categorical(
                                [[not slip, slip]])
            elif self.config['label_dimension'] == 'y':
                slip = abs(self.__get_slip_dir(idx)[1]) > self.config['slip_thresh']['speed']
                return self.booleans_to_categorical(
                                [[not slip, slip]])
            else:
                raise ValueError("Invalid label dimension {}".format(self.config['label_dimension']))
        else:
            raise ValueError("Invalid label type {}".format(self.config['label_type']))


    def __get_pressure(self, idx):
        return (self.__data['pressure'][idx] + PRESSURE_OFFSET)/PRESSURE_SCALE

    def __get_mode(self):
        return self.__data['mode']

    def __get_temp(self, idx):
        return self.__data['temp'][idx]

    def __get_time(self, idx):
        return self.__data['time'][idx]

    def __get_slip_dir(self, idx):
        if self.__get_mode() == FLOW_MODE:
            return self.__data['slip'][idx, 1:3]
        else:
            return self.__data['slip'][idx, 0:2] * SPEED_SCALE

    def __get_slip_angle(self, idx):
        return self.__slip_angle_data[idx]

    def __get_slip_speed(self, idx):
        if self.__get_mode() == FLOW_MODE:
            return self.__data['slip'][idx, 0]
        else:
            if isinstance(idx, (int, long)):
                return np.linalg.norm(self.__data['slip'][idx, 0:2])
            return np.linalg.norm(self.__data['slip'][idx, 0:2], axis=1)

    def __get_ang_vel(self, idx):
        if self.__get_mode() != FLOW_MODE:
            return self.__data['slip'][idx, 2] * ANG_SPEED_SCALE
        else:
            raise self.__data['slip'][idx, 2] * 0   #ZEROS

    def __get_slip_std(self, idx):
        if self.__get_mode() == FLOW_MODE:
            return self.__data['slip'][idx, 3:5]
        else:
            raise NotImplementedError


if __name__ == "__main__":
    config = load_yaml("../configs/base_config.yaml")
    config = config['data']
    data_base = config['data_home']
    data_dirs = [os.path.join(data_base, o) for o in os.listdir(data_base)
                if os.path.isdir(os.path.join(data_base,o)) and "takktile_" in o ]
    dataloaders = []
    for dir in data_dirs:
        dataloaders.append(takktile_dataloader(dir, config=config))
        if not dataloaders[-1].empty():
            print(dataloaders[-1].get_data_mean())
            print(dataloaders[-1].get_data_std())
            print(dataloaders[-1].get_data_min())
            print(dataloaders[-1].get_data_max())
            # print(dataloaders[-1].get_slip_idx())
            print(dataloaders[-1].booleans_to_categorical([[False, False, True], [False, True, False]]))
            print(dataloaders[-1].booleans_to_categorical([[False, True, True], [False, True, False]]))