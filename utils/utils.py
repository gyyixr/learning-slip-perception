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
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.fftpack import rfft, fft

def fft_real(array ,axis=-1):
    assert len(np.shape(array)) > 0
    assert -len(np.shape(array)) <= axis < len(np.shape(array))
    assert isinstance(axis, int)

    return abs(fft(x=array, axis=axis))

if __name__ == "__main__":
    a = np.random.random([10,32,21])
    print(a[0,0,:])
    print(fft_real(a, axis=-1)[0,0,:])
