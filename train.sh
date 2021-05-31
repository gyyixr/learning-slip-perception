#!/usr/bin/env bash

train_epochs () {
    ##############################################
    # Train a model with a pregenerated model folder
    # Params $1(int|number of epochs) $2(relative/abs path to yaml config file)
    # If $1 < trained epochs, not training will occurs
    ##############################################

    SEARCH_STR="epochs: 0"
    REPLACE_STR="epochs: $1"
    sed -i "s/${SEARCH_STR}/${REPLACE_STR}/" $2

    python2.7 main.py $2

    REPLACE_STR="epochs: 0"
    SEARCH_STR="epochs: $1"
    sed -i "s/${SEARCH_STR}/${REPLACE_STR}/" $2
}

test_net ()  {
    ##############################################
    # Test an already trained network using config file
    # Params $1(relative/abs location of yaml config file)
    ##############################################

    SEARCH_STR="use_best_model: false"
    REPLACE_STR="use_best_model: true"
    sed -i "s/${SEARCH_STR}/${REPLACE_STR}/" $1

    SEARCH_STR="save_last_model: true"
    REPLACE_STR="save_last_model: false"
    sed -i "s/${SEARCH_STR}/${REPLACE_STR}/" $1

    SEARCH_STR="balance_data: true"
    REPLACE_STR="balance_data: false"
    sed -i "s/${SEARCH_STR}/${REPLACE_STR}/" $1

    train_epochs 0 $1


    SEARCH_STR="balance_data: false"
    REPLACE_STR="balance_data: true"
    sed -i "s/${SEARCH_STR}/${REPLACE_STR}/" $1

    REPLACE_STR="use_best_model: false"
    SEARCH_STR="use_best_model: true"
    sed -i "s/${SEARCH_STR}/${REPLACE_STR}/" $1

    REPLACE_STR="save_last_model: true"
    SEARCH_STR="save_last_model: false"
    sed -i "s/${SEARCH_STR}/${REPLACE_STR}/" $1
}


## SLIP EXPERIMENTS
# train_epochs 800 ./logs/models/TCN_20210210-011650/config.yaml # plastic + sphere + cylinder(both) + rotation + static + data balancing (changed labelling)
# train_epochs 800 ./logs/models/FREQ_20210212-011047/config.yaml # plastic + sphere + cylinder(both) + rotation + static + data balancing (changed labelling) (100)

test_net ./logs/models/TCN_20210210-011650/config.yaml # plastic + sphere + cylinder(both) + rotation + static + data balancing (changed labelling)
test_net ./logs/models/FREQ_20210212-011047/config.yaml # plastic + sphere + cylinder(both) + rotation + static + data balancing (changed labelling) (100)


## DIRECTION EXPERIMENTS
# train_epochs 800 ./logs/models/TCN_20210107-123615/config.yaml # Translation only
# train_epochs 800 ./logs/models/TCN_20210203-005213/config.yaml # plastic + sphere + cylinder(both) - rotation - release
