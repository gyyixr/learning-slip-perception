#!/usr/bin/env python2.7

"""
DataAugment.py

Data augmentation helper file

Developed at UTIAS, Toronto.

author: Abhinav Grover

date: Nov 17, 2020
"""

##################### Error printing
from __future__ import print_function
import sys

def eprint(*args, **kwargs):
    print("ERROR: {}: ".format(__file__))
    print(*args, file=sys.stderr, **kwargs)
#####################

import numpy as np

# CONSTANTS
DO_NOTHING = 'none'
HORIZONTAL_FLIP = 'h_flip'
VERTICAL_FLIP = 'f_flip'
BOTH_FLIP = 'both_flip'
GAUSS_NOISE = 'g_noise'

VEL_X = 0
VEL_Y = 1

# SENSOR SCHEMATIC
#     _________
#    /         \
#   / 5   4   3 \
#  |             |
#  |  0   1   2  |
#  ---------------
# Y
# ^
# |
# |
# |
# |
#  ---------------->  X

class takktile_data_augment():
    def __init__(self, data_config):
        """
        takktile data augmentation class. Produces random horizontal and vertical flips in the data.
        """
        if not data_config or not 'augment' in data_config:
            self.config = {'mode': DO_NOTHING, 'probability': 0.0}
        else:
            self.config = data_config['augment']
        
        assert 'mode' in self.config
        assert 'probability' in self.config and 0.0 <= self.config['probability'] <= 1.0

    def __call__(self, data):
        """
        Augmentation API function

        input
        -----------
        data : x,y data with x being network input (size of series length) and y being the label
        """

        if np.random.uniform() <= self.config['probability']:
            if self.config['mode'] == HORIZONTAL_FLIP:
                return self.__horizontal_flip(data)
            elif self.config['mode'] == VERTICAL_FLIP:
                return self.__vertical_flip(data)
            elif self.config['mode'] == BOTH_FLIP:
                return self.__both_flip(data)
            elif self.config['mode'] == GAUSS_NOISE:
                return self.__gaussian_noise(data)
            else:
                return self.__do_nothing(data)
        else:
            return self.__do_nothing(data)

    def __do_nothing(self, data):
        """
        The do nothing augmentation | Used to disable any augmentation

        input
        -----------
        data : x,y data with x being network input and y being the velocities
        """
        return data

    def __horizontal_flip(self, data):
        """
        Horizontally mirror both inputs and outputs

        input
        -----------
        data : x,y data with x being network input and y being the velocities
        """
        x, y, z = data
        if x.shape[1] != 6:
            eprint("Cannot Augment input data of shape {}".format(x.shape))
            return data

        x = np.flip(x, axis=-1) # flip inputs
        y[VEL_Y] *= -1
        z *= -1
        return x, y, z

    def __vertical_flip(self, data):
        """
        Vertically mirror both inputs and outputs

        input
        -----------
        data : x,y data with x being network input and y being the velocities
        """
        x, y, z = data
        if x.shape[1] != 6:
            eprint("Cannot Augment input data of shape {}".format(x.shape))
            return data

        x = x[:, [2,1,0,5,4,3]] # flip inputs
        y[VEL_X] *= -1
        z *= -1
        return x, y, z

    def __both_flip(self, data):
        """
        Both Horizontally and vertically mirror both inputs and outputs

        input
        -----------
        data : x,y data with x being network input and y being the velocities
        """
        x, y, z = data
        if x.shape[1] != 6:
            eprint("Cannot Augment input data of shape {}".format(x.shape))
            return data

        prob = np.random.uniform()
        if prob <= 1/3:
            x, y, z = self.__horizontal_flip((x, y, z))
        elif prob <= 2/3:
            x, y, z = self.__vertical_flip((x, y, z))
        else:
            x, y, z = self.__horizontal_flip((x, y, z))
            x, y, z = self.__vertical_flip((x, y, z))

        return x, y, z

    def __gaussian_noise(self, data):
        """
        Add gaussian noise to inputs only

        input
        -----------
        data : x,y data with x being network input and y being the velocities
        """
        eprint("Gaussian Noise Not implemented yet")
        return self.__do_nothing(data)