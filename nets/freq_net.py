#!/usr/bin/env python2.7

import inspect
from typing import List
import copy
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K, Sequential, optimizers
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import Conv2D, Dense, Flatten, BatchNormalization, Dropout, SpatialDropout2D

# Fix for CUDNN failed to init: https://github.com/tensorflow/tensorflow/issues/24828#issuecomment-464910864
import tensorflow.compat.v1 as tf1

config = tf1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf1.Session(config=config)


def freq_model(input_shape,
               num_classes,
               cnn_filters_num,
               dense_layer_num,
               batch_norm=True,
               dropout_rate=0.1,
               kernel_initializer="he_normal",
               padding="same",
               lr=0.002,
               activation = "relu"):

    assert (isinstance(cnn_filters_num, list) or isinstance(cnn_filters_num, tuple)) and len(cnn_filters_num) > 0
    assert (isinstance(dense_layer_num, list) or isinstance(dense_layer_num, tuple)) and len(dense_layer_num) > 0

    model = Sequential()
    if dropout_rate > 0.0:
        model.add(Dropout(dropout_rate, name="dropout_input"))

    # CNN
    for i, f in enumerate(cnn_filters_num):
        name = "Conv_2d_{}".format(i)
        if i == 0:
            model.add(Conv2D(filters=f,
                             kernel_size=(3, 3),
                             activation=activation,
                             name=name,
                             kernel_initializer=kernel_initializer,
                             padding=padding,
                             input_shape=input_shape))
        elif i == len(cnn_filters_num)-1:
            model.add(Conv2D(filters=f,
                             kernel_size=(2, 2),
                             activation=activation,
                             name=name,
                             kernel_initializer=kernel_initializer,
                             padding="valid",
                             input_shape=input_shape))
        else:
            model.add(Conv2D(filters=f,
                             kernel_size=(3, 3),
                             activation=activation,
                             name=name,
                             padding=padding,
                             kernel_initializer=kernel_initializer))
        if dropout_rate > 0.0:
            model.add(SpatialDropout2D(dropout_rate, name="spacial_dropout_{}".format(i)))
    model.add(Flatten())

    # MLP
    for j,d in enumerate(dense_layer_num):
        name = "Dense_{}".format(j)
        model.add(Dense(units=d,
                        activation=activation,
                        name=name,
                        kernel_initializer=kernel_initializer))
        if dropout_rate > 0.0:
            model.add(Dropout(dropout_rate, name="Dropout_dense_{}".format(j)))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(optimizer=optimizers.Adam(lr=lr, clipnorm=1.),
                  metrics=['categorical_accuracy'],
                  loss= CategoricalCrossentropy(from_logits=False))

    return model

if __name__ == "__main__":
    model = freq_model(input_shape=(2,3,10),
                        num_classes=2,
                        cnn_filters_num=[20,20],
                        dense_layer_num=(40,20),
                        batch_norm=True,
                        dropout_rate=0.1,
                        kernel_initializer="he_normal",
                        padding="same",
                        lr=0.002,
                        activation = "relu")
    model.build((32, 2,3,10))
    model.summary()
