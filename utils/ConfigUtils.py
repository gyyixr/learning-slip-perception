#!/usr/bin/env python2.7

"""
ConfigUtils.py

Util file for handling takktile recorded data

Developed at UTIAS, Toronto.

author: Abhinav Grover

Date: September 23, 2020
"""

##################### Error printing
from __future__ import print_function
import sys

def eprint(*args, **kwargs):
    print("ERROR: {}: ".format(__file__))
    print(*args, file=sys.stderr, **kwargs)
#####################

# Standard Imports
import yaml
import os
CWD = os.path.dirname(os.path.realpath(__file__))

def load_yaml(yaml_file):
    if not file:
        eprint("YAML file not provided: {}".format(yaml_file))
        return

    if not os.path.isfile(yaml_file):
        eprint("YAML file does not exists: {}".format(yaml_file))
        raise ValueError

    if not '.yaml' in yaml_file:
        eprint("YAML file has the wrong format: {}".format(yaml_file))
        raise ValueError

    input_stream = file(yaml_file, 'r')
    config = yaml.load(input_stream)

    if not is_config_valid(config):
        raise ValueError("Invalid Config")

    return config

def is_config_valid(base_config):
    if not isinstance(base_config, dict):
        eprint("Provided config is not a dictionary")
        return False

    if 'data' not in base_config:
        eprint("Provided config doesnt contain \'data\' key")
        return False
    data_config = base_config['data']

    if 'net' not in base_config:
        eprint("Provided config doesnt contain \'net\' key")
        return False
    net_config = base_config['net']

    if 'training' not in base_config:
        eprint("Provided config doesnt contain \'training\' key")
        return False
    training_config = base_config['training']

    ###
    # CONFIG RULES
    ###

    # 1. lable_type 'value' must be used with regression, while others cannot be used with regression
    if (data_config['label_type'] == 'value') != (training_config['regression'] == True):
        eprint(" lable_type \'value\' must be used with regression, while others cannot be used with regression")
        return False

    return True


def save_yaml(dict_, yaml_file):
    if not '.yaml' in yaml_file:
        eprint("YAML file has the wrong format: {}".format(yaml_file))
        raise ValueError

    if not isinstance(dict_, dict):
        eprint("YAML dict is not vali: {}".format(dict_))
        raise ValueError

    output_stream = file(yaml_file, 'w')
    return yaml.dump(dict_, output_stream)


if __name__ == "__main__":
    # Test Loading
    config = load_yaml(CWD + '/test.yaml')
    section1 = config['section1']

    # Test Types
    print(section1['param1'])
    print(section1['param2']['param3'] == False)
    print(section1['list_float'])
    print(section1['list_int'])
    print(section1['list'])

    # Test saving new file
    import numpy as np
    section1['param2']['new_param'] = ([0.1, 0.2], [9.0])
    save_yaml(config, './new_test.yaml')
    new_config = load_yaml('./new_test.yaml')
    print(new_config['section1'])
