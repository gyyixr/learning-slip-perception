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
    return yaml.load(input_stream)

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
    config = load_yaml('./test.yaml')
    section1 = config['section1']
    
    # Test Types
    print(section1['param1'])
    print(section1['param2'])
    print(section1['list_float'])
    print(section1['list_int'])
    print(section1['list'])

    # Test saving new file
    section1['new_param'] = ['1', '2', '3']
    save_yaml(config, './new_test.yaml')
    new_config = load_yaml('./new_test.yaml')
    print(new_config['section1'])
