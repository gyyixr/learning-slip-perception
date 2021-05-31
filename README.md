# Learning Slip Perception
This repository contains python scripts to train a slip detection Temporal Convolution Network ([TCN](https://github.com/philipperemy/keras-tcn)) using the data collected from the [Takktile sensor kit](https://www.labs.righthandrobotics.com/robotiq-kit). Details about this work can be found in this paper: [Under Pressure: Learning to Detect Slip with Barometric Tactile Sensors](https://papers.starslab.ca/slip-detection/). 

Table of Content
- [Learning Slip Perception](#learning-slip-perception)
- [Setup](#setup)
  - [Pre-requisites](#pre-requisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Training a New Model](#training-a-new-model)
  - [Monitoring Training with Tensorboard](#monitoring-training-with-tensorboard)
- [Software Architecture](#software-architecture)

# Setup

## Pre-requisites
The training scripts are all written in Python2, in order to ensure compatibility with ROS Kinetic. The authors use `Keras` for building and training the TCN models. 

* Install TensorFlow 2.1.0 (CPU or GPU) by following the instructions [here](https://www.tensorflow.org/install/pip). `Keras` is now packaged as part of TensorFlow framework. This is last TensorFlow version that supports Python2.

* [Email](mailto:abhinav.grover@robotics.utias.utoronto.ca) the author to request access to the *Takktile* training dataset.

## Installation
* Clone this repository
* ```shell
  cd <path to learning-slip-perception>
  ```
* ```shell
  git submodule init && git submodule update
  ```

# Usage
The main script takes a yaml configuration file as input, which provides values for all the data, network, and training parameters. The base config files are present in the [configs](/configs) directory.

## Training a New Model
* Modify the base config files to create a training experiment. Modify the `data_home` parameter and replace it with the path to your *Takktile* data folder. You can use the `epochs` parameter to controls the number of epochs.
* ```shell
  python2.7 main.py ./configs/<file name>.yaml
  ```
  replace the name with that of the config file you edited. 
* A new folder will be created with in the [models directory](/logs/models).

## Continue Training of an Exisiting Model
* Locate the model you would like to continue training. (A copy of the config file is saved with each model in the [models directory](/logs/models))
* Modify the `epochs` parameter to indicate the number of additional epochs you would like to train.
* ```
  python2.7 main.py ./logs/models/<model_name>/config.yaml
  ```
  replace the name with that of the config file you edited. No new directory will be created.
* You can use the bash script, `./train.sh`, to automate training experiments.

## Monitoring Training with Tensorboard
The tensorboard scalars are saves in the [scalars directory](/logs/scalars). 

```shell
tensorboard --logdir ./logs/scalars/<scaler_dir>
```

# Software Architecture
TODO
