#!/usr/bin/env bash

train_100_epochs () {
    SEARCH_STR="epochs: 0"
    REPLACE_STR="epochs: 100"
    sed -i "s/${SEARCH_STR}/${REPLACE_STR}/" $1

    python2.7 train.py $1
}

## X Baselines
# train_100_epochs ./logs/models/TCN_20201014-091533/config.yaml # x | slip      | 0.1 | all
# train_100_epochs ./logs/models/TCN_20201014-091617/config.yaml # x | direction | 0.1 | all
# train_100_epochs ./logs/models/TCN_20201014-091659/config.yaml # x | value     | 0.1 | all

## Y baselines
# train_100_epochs ./logs/models/TCN_20201014-091830/config.yaml # y | slip      | 0.1 | all
# train_100_epochs ./logs/models/TCN_20201014-092005/config.yaml # y | direction | 0.1 | all
# train_100_epochs ./logs/models/TCN_20201014-091750/config.yaml # y | value     | 0.1 | all

## X slip Experiments
# train_100_epochs ./logs/models/TCN_20201014-131550/config.yaml # x | slip      | 0.2 | all 
# train_100_epochs ./logs/models/TCN_20201014-131914/config.yaml # x | slip      | 0.1 | all | low_complex
# train_100_epochs ./logs/models/TCN_20201014-154553/config.yaml # x | slip      | 0.1 | all | more_regular
# train_100_epochs ./logs/models/TCN_20201014-171558/config.yaml # x | slip      | 0.1 | all | x,y,combined
# train_100_epochs ./logs/models/TCN_20201014-180111/config.yaml # x | slip      | 0.1 | all | x,y
# train_100_epochs ./logs/models/TCN_20201015-110348/config.yaml # x | slip      | 0.1 | all | low_complexity | more_regular | x,y
# train_100_epochs ./logs/models/TCN_20201015-121948/config.yaml # x | slip      | 0.1 | all | low_complexity | more_regular | x,combined
# train_100_epochs ./logs/models/TCN_20201015-130925/config.yaml # x | slip      | 0.1 | all | low_complexity | more_regular
# train_100_epochs ./logs/models/TCN_20201015-134320/config.yaml # x | slip      | 0.08| all | low_complexity | more_regular
# train_100_epochs ./logs/models/TCN_20201015-135924/config.yaml # x | slip      | 0.12| all | low_complexity | more_regular
# train_100_epochs ./logs/models/TCN_20201015-142020/config.yaml # x | slip      | 0.15| all | low_complexity | more_regular
train_100_epochs ./logs/models/TCN_20201015-183519/config.yaml # x | slip      | 0.135| all | low_complexity | more_regular

## X direction Experiments
# train_100_epochs ./logs/models/TCN_20201014-142345/config.yaml # x | direction | 0.1 | slip