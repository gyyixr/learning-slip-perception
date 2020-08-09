#!/usr/bin/env python2.7

"""
DataGenerator.py

Util file for handling takktile recorded data

Developed at UTIAS, Toronto.

author: Abhinav Grover
"""

##################### Error printing
from __future__ import print_function
import sys

def eprint(*args, **kwargs):
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

class takktile_datagenerator(tf.keras.utils.Sequence):

    def __init__(self, df, x_col, y_col=None, batch_size=32, num_classes=None, shuffle=True):
        self.batch_size = batch_size
        self.df = df
        self.indices = [] #self.df.index.tolist()
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.x_col = x_col
        self.y_col = y_col
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices // self.batch_size)

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]
        
        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        X = None # logic
        y = None # logic
        
        # for i, id in enumerate(batch):
        #     X[i,] = None # logic
        #     y[i] = None  # labels

        return X, y

if __name__ == "__main__":
    dg = takktile_datagenerator(None, None)