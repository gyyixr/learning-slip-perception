#!/usr/bin/env bash

train_epochs () {
    SEARCH_STR="epochs: 0"
    REPLACE_STR="epochs: $1"
    sed -i "s/${SEARCH_STR}/${REPLACE_STR}/" $2

    python2.7 train.py $2

    REPLACE_STR="epochs: 0"
    SEARCH_STR="epochs: $1"
    sed -i "s/${SEARCH_STR}/${REPLACE_STR}/" $2
}

train_100_epochs () {
    train_epochs 100 $1
}

test_net ()  {
    SEARCH_STR="use_best_model: false"
    REPLACE_STR="use_best_model: true"
    sed -i "s/${SEARCH_STR}/${REPLACE_STR}/" $1

    SEARCH_STR="save_last_model: true"
    REPLACE_STR="save_last_model: false"
    sed -i "s/${SEARCH_STR}/${REPLACE_STR}/" $1

    train_epochs 0 $1

    REPLACE_STR="use_best_model: false"
    SEARCH_STR="use_best_model: true"
    sed -i "s/${SEARCH_STR}/${REPLACE_STR}/" $1

    REPLACE_STR="save_last_model: true"
    SEARCH_STR="save_last_model: false"
    sed -i "s/${SEARCH_STR}/${REPLACE_STR}/" $1
}


## X Baselines
# train_100_epochs ./logs/saved_scalers/TCN_20201014-091533/config.yaml # x | slip      | 0.1 | all
# train_100_epochs ./logs/saved_scalers/TCN_20201014-091617/config.yaml # x | direction | 0.1 | all
# train_100_epochs ./logs/saved_scalers/TCN_20201014-091659/config.yaml # x | value     | 0.1 | all

## Y baselines
# train_100_epochs ./logs/saved_scalers/TCN_20201014-091830/config.yaml # y | slip      | 0.1 | all
# train_100_epochs ./logs/saved_scalers/TCN_20201014-092005/config.yaml # y | direction | 0.1 | all
# train_100_epochs ./logs/saved_scalers/TCN_20201014-091750/config.yaml # y | value     | 0.1 | all


## X slip Experiments [OLD]
# train_100_epochs ./logs/saved_scalers/TCN_20201014-131550/config.yaml # x | slip      | 0.2 | all
# train_100_epochs ./logs/saved_scalers/TCN_20201014-131914/config.yaml # x | slip      | 0.1 | all | low_complex
# train_100_epochs ./logs/saved_scalers/TCN_20201014-154553/config.yaml # x | slip      | 0.1 | all | more_regular
# train_100_epochs ./logs/saved_scalers/TCN_20201014-171558/config.yaml # x | slip      | 0.1 | all | x,y,combined
# train_100_epochs ./logs/saved_scalers/TCN_20201014-180111/config.yaml # x | slip      | 0.1 | all | x,y
# train_100_epochs ./logs/saved_scalers/TCN_20201015-110348/config.yaml # x | slip      | 0.1 | all | low_complexity | more_regular | x,y
# train_100_epochs ./logs/saved_scalers/TCN_20201015-121948/config.yaml # x | slip      | 0.1 | all | low_complexity | more_regular | x,combined
# train_100_epochs ./logs/saved_scalers/TCN_20201015-130925/config.yaml # x | slip      | 0.1 | all | low_complexity | more_regular
# train_100_epochs ./logs/saved_scalers/TCN_20201015-134320/config.yaml # x | slip      | 0.08| all | low_complexity | more_regular
# train_100_epochs ./logs/saved_scalers/TCN_20201015-135924/config.yaml # x | slip      | 0.12| all | low_complexity | more_regular
# train_100_epochs ./logs/saved_scalers/TCN_20201015-142020/config.yaml # x | slip      | 0.15| all | low_complexity | more_regular
# train_100_epochs ./logs/saved_scalers/TCN_20201015-183519/config.yaml # x | slip      | 0.135| all | low_complexity | more_regular
# train_100_epochs ./logs/saved_scalers/TCN_20201019-110150/config.yaml # x | slip      | 0.1  | all | KL
# train_100_epochs ./logs/saved_scalers/TCN_20201019-112026/config.yaml # x | slip      | 0.1  | all | CCE
# train_100_epochs ./logs/saved_scalers/TCN_20201019-112723/config.yaml # x | slip      | 0.1  | all | CCE | BN
# train_100_epochs ./logs/saved_scalers/TCN_20201019-113035/config.yaml # x | slip      | 0.1  | all | CCE | LN

## X direction Experiments [OLD]
# train_100_epochs ./logs/saved_scalers/TCN_20201014-142345/config.yaml # x | direction | 0.1 | slip
# train_epochs 100 ./logs/saved_scalers/TCN_20201019-194539/config.yaml # x | direction | 0.12 | all | v3
# train_epochs 100 ./logs/saved_scalers/TCN_20201019-194624/config.yaml # x | direction | 0.12 | all | v1-3
# train_epochs 100 ./logs/saved_scalers/TCN_20201019-222746/config.yaml # x | direction | 0.12 | all | v3 x,y |
# train_epochs 100 ./logs/saved_scalers/TCN_20201019-223353/config.yaml # x | direction | 0.12 | all | v1-3 | larger_network
# train_epochs 100 ./logs/saved_scalers/TCN_20201020-103033/config.yaml # x | direction | 0.12 | all | v3 x,y | larger_network

## X direction Tests [OLD]
# test_net ./logs/saved_scalers/TCN_20201019-194539/config.yaml # x | direction | 0.12 | all | v3
# test_net ./logs/saved_scalers/TCN_20201019-194624/config.yaml # x | direction | 0.12 | all | v1-3
# test_net ./logs/saved_scalers/TCN_20201019-222746/config.yaml # x | direction | 0.12 | all | v3 x,y |
# test_net ./logs/saved_scalers/TCN_20201019-223353/config.yaml # x | direction | 0.12 | all | v1-3 | larger_network
# test_net ./logs/saved_scalers/TCN_20201020-103033/config.yaml # x | direction | 0.12 | all | v3 x,y | larger_network

## X Value Experiments
# train_epochs 200 ./logs/saved_scalers/TCN_20201018-002830/config.yaml # x | value | all |
# test_net ./logs/saved_scalers/TCN_20201018-002830/config.yaml # x | value | all |

## X slip Experiments [NEW]
# train_epochs 100 ./logs/models/TCN_20201104-094448/config.yaml # 0.05 | all | v3 x | reg_net | [24, 24]

## X slip Tests [NEW]
# test_net ./logs/models/TCN_20201104-094448/config.yaml # 0.05 | all | v3 x | reg_net | [24, 24]

## X direction Experiments [NEW]
# train_epochs 100 ./logs/models/TCN_20201104-090139/config.yaml # 0.05 | all | v3 x | reg_net | [24, 24]


## Translation Slip Experiments [OLD]
# train_100_epochs ./logs/saved_scalers/TCN_20201016-200454_0/config.yaml # translation | slip      | 0.1  | all | low_complexity | more_regular | LN
# train_100_epochs ./logs/saved_scalers/TCN_20201016-200753_0/config.yaml # translation | slip      | 0.1  | all | combined | low_complexity | more_regular | LN
# train_epochs 200 ./logs/saved_scalers/TCN_20201016-222905/config.yaml # translation | slip      | 0.1  | all | combined | low_complexity | more_regular | LN
# train_epochs 100 ./logs/saved_scalers/TCN_20201017-123626/config.yaml # translation | slip      | 0.1  | all | combined | low_complexity | more_regular | LN
# train_epochs 100 ./logs/saved_scalers/TCN_20201017-123827/config.yaml # translation | slip      | 0.1  | all | combined | low_complexity | more_regular | LN
# train_epochs 100 ./logs/saved_scalers/TCN_20201018-160133/config.yaml # translation | slip      | 0.12  | GELU
# train_epochs 100 ./logs/saved_scalers/TCN_20201018-161148/config.yaml # translation | slip      | 0.12  | SELU
# train_epochs 100 ./logs/saved_scalers/TCN_20201018-233841/config.yaml # translation | slip      | 0.12  | full SELU

## Translation Slip Tests [OLD]
# test_net ./logs/saved_scalers/TCN_20201017-123626_100/config.yaml # translation | slip      | 0.1  | all | combined | low_complexity | more_regular | LN
# test_net ./logs/saved_scalers/TCN_20201017-123827_100/config.yaml # translation | slip      | 0.1  | all | combined | low_complexity | more_regular | LN
# test_net ./logs/saved_scalers/TCN_20201018-160133/config.yaml # translation | slip      | 0.12  | GELU
# test_net ./logs/saved_scalers/TCN_20201018-161148/config.yaml # translation | slip      | 0.12  | SELU
# test_net ./logs/saved_scalers/TCN_20201018-233841/config.yaml # translation | slip      | 0.12  | full SELU

## Translation Slip Experiments [NEW]
# train_100_epochs ./logs/saved_scalers/TCN_20201029-153721/config.yaml # v3 | simple_net | 0.5
# train_epochs 200 ./logs/models/TCN_20201102-103832/config.yaml # v3 | simple_net | 0.05 | lr0.002 | [24, 24] | combined

#AUG
# train_epochs 400 ./logs/models/TCN_20201130-182354/config.yaml # v3 | all_data | matt only | both_flip 0.75
# test_net ./logs/models/TCN_20201130-204718/config.yaml # v3 | all_data | matt only | both_flip 0.75 | slip_v 0.02
# test_net ./logs/models/TCN_20201130-234547/config.yaml # v3 | all_data | matt only | both_flip 0.75 | slip_v 0.02 | early_stop | class_weight
# train_epochs 400 ./logs/models/TCN_20201201-162803/config.yaml # v3 | all_data | matt only | both_flip 0.75 | gauss 2.0
# train_epochs 400 ./logs/models/TCN_20201201-170034/config.yaml # v3 | all_data | matt only | both_flip 0.75 | gauss 5.0
# test_net ./logs/models/TCN_20201208-185532/config.yaml # v3 | all_data | all materials | both_flip 0.75 | gauss 3.0
# test_net ./logs/models/TCN_20201209-212419/config.yaml # v3 | all_data | all materials | both_flip 0.75 | gauss 3.0 | 0.05
# test_net ./logs/models/TCN_20201222-013042/config.yaml # v3 | all_data | all materials | both_flip 0.75 | gauss 3.0 | 0.05
# train_epochs 200 ./logs/models/TCN_20201223-235045/config.yaml # v3 | all_data | matt and felt | both_flip 0.75 | gauss 5.0 | 0.03
train_epochs 200 ./logs/models/TCN_20201224-160639/config.yaml # v3 | all_data | matt and felt | both_flip 0.75 | gauss 3.0 | 0.02

#AUG+BALANCING
# test_net ./logs/models/TCN_20201206-214910/config.yaml # v3 | all_data | matt only | both_flip 0.5 | gauss 1.0
# test_net ./logs/models/TCN_20201209-030216/config.yaml # v3 | all_data | all material| both_flip 0.75 | gauss 3.0 | 0.03 | balancing

## Translation Direction Experiments [OLD]
# train_epochs 100 ./logs/saved_scalers/TCN_20201021-160416/config.yaml # translation | direction | v1-3 | simple_net
# train_epochs 100 ./logs/saved_scalers/TCN_20201021-160323/config.yaml # translation | direction | v1-3 | complex_net | more drop
# train_epochs 100 ./logs/saved_scalers/TCN_20201023-130019/config.yaml # translation | direction | v1-3 | class_weights | simple_net
# train_epochs 100 ./logs/saved_scalers/TCN_20201023-130123/config.yaml # translation | direction | v1-3 | class_weights | simple_net + larger kernel
# train_epochs 100 ./logs/saved_scalers/TCN_20201026-173847/config.yaml # translation | direction | v3 | simple_net
# train_epochs 200 ./logs/saved_scalers/TCN_20201026-222852/config.yaml # translation | direction | v1-3 | 1,1,2,2,2
# train_epochs 200 ./logs/saved_scalers/TCN_20201026-223615/config.yaml # translation | direction | v1-3 | 1,1,1,1,1

## Translation Direction Tests [OLD]
# test_net ./logs/saved_scalers/TCN_20201021-160416/config.yaml # translation | direction | v1-3 | simple_net
# test_net ./logs/saved_scalers/TCN_20201021-160323/config.yaml # translation | direction | v1-3 | complex_net | more drop
# test_net ./logs/saved_scalers/TCN_20201023-130019/config.yaml # translation | direction | v1-3 | class_weights | simple_net
# test_net ./logs/saved_scalers/TCN_20201023-130123/config.yaml # translation | direction | v1-3 | class_weights | simple_net + larger kernel
# test_net ./logs/saved_scalers/TCN_20201026-173847/config.yaml # translation | direction | v1-3 | class_weights | simple_net + larger kernel
# test_net ./logs/saved_scalers/TCN_20201026-222852/config.yaml # translation | direction | v1-3 | 1,1,2,2,2
# test_net ./logs/saved_scalers/TCN_20201026-223615/config.yaml # translation | direction | v1-3 | 1,1,1,1,1

## Translation Direction Experiments [NEW]
# train_epochs 300 ./logs/saved_scalers/TCN_20201029-154119/config.yaml # v3 | simple_net | 0.05
# train_epochs 200 ./logs/saved_scalers/TCN_20201030-113239/config.yaml # v3 | simple_net | 0.05 | lr0.002 | [24, 24]
# train_epochs 200 ./logs/saved_scalers/TCN_20201030-113715/config.yaml # v3 | simple_net | 0.05 | lr0.002 | filters 12
# train_epochs 200 ./logs/saved_scalers/TCN_20201030-194305/config.yaml # v3 | simple_net | 0.05 | lr0.002 | [48, 24, 24]
# train_epochs 200 ./logs/models/TCN_20201102-113640/config.yaml # v3 | simple_net | 0.05 | lr0.002 | [24, 24] | combined
# train_epochs 200 ./logs/models/TCN_20201108-165653/config.yaml # v3 | selu->norm
# train_epochs 100 ./logs/models/TCN_20201108-172039/config.yaml # v3 | selu->norm | [12, 24, 36, 48] | no_skip_conn
# train_epochs 400 ./logs/models/TCN_20201118-092102/config.yaml # v3 | slip_data | matt only
# train_epochs 100 ./logs/models/TCN_20201118-092228/config.yaml # v3 | slip_data | matt and felt
# train_epochs 100 ./logs/models/TCN_20201118-141530/config.yaml # v3 | slip_data | matt only | delta
# train_epochs 100 ./logs/models/TCN_20201118-151840/config.yaml # v3 | slip_data | matt only | truncate 70
# train_epochs 100 ./logs/models/TCN_20201118-160527/config.yaml # v3 | slip_data | matt only | delta & truncate 70
# train_epochs 200 ./logs/models/TCN_20201118-170350/config.yaml # v3 | slip_data | matt only | normal+delta

#AUG+Balancing
# train_epochs 200 ./logs/models/TCN_20201209-045919/config.yaml # v3 | slip_data | all material | both_flip 0.75 | gauss 3.0
# test_net ./logs/models/TCN_20201209-130814/config.yaml # v3 | all_data | felt only | both_flip 0.5 | gauss 3.0

#AUG
# train_epochs 400 ./logs/models/TCN_20201119-224913/config.yaml # v3 | slip_data | matt only | h_flip 0.5
# train_epochs 400 ./logs/models/TCN_20201119-225001/config.yaml # v3 | slip_data | matt only | v_flip 0.5
# train_epochs 400 ./logs/models/TCN_20201119-225018/config.yaml # v3 | slip_data | matt only | both_flip 0.5
# train_epochs 400 ./logs/models/TCN_20201201-170250/config.yaml # v3 | slip_data | matt only | both_flip 0.75 | gauss 2.0
# train_epochs 400 ./logs/models/TCN_20201201-225311/config.yaml # v3 | slip_data | matt only | both_flip 0.75 | gauss 0.0
# train_epochs 162 ./logs/models/TCN_20201202-102603/config.yaml # v3 | slip_data | matt only | both_flip-V 0.5 | gauss 1.0
# train_epochs 98 ./logs/models/TCN_20201202-104746/config.yaml # v3 | slip_data | matt only | both_flip 0.5 | gauss 3.0
# train_epochs 400 ./logs/models/TCN_20201202-112234/config.yaml # v3 | slip_data | matt only | both_flip-V 0.5 | gauss 3.0
# train_epochs 200 ./logs/models/TCN_20201209-045658/config.yaml # v3 | slip_data | all material | both_flip 0.75 | gauss 3.0
# train_epochs 200 ./logs/models/TCN_20201209-130716/config.yaml # v3 | all_data | felt only | both_flip 0.5 | gauss 3.0 | 0.03
# test_net ./logs/models/TCN_20201209-212200/config.yaml # v3 | slip_data | all material | both_flip 0.5 | gauss 3.0 | 0.05
# train_epochs 1000 ./logs/models/TCN_20201210-044916/config.yaml # v3 | slip_data | all material | both_flip 0.5 | gauss 0.0 | 0.05
# test_net ./logs/models/TCN_20201222-013544/config.yaml # v3 | slip_data | all material | both_flip 0.5 | gauss 0.0 | 0.05
# train_epochs 1000 ./logs/models/TCN_20201223-234846/config.yaml # v3 | all_data | matt and felt | both_flip 0.5 | gauss 3.0 | 0.03

## Translation Value Experiments
# train_epochs 200 ./logs/saved_scalers/TCN_20201018-003156/config.yaml # translation | value | all |
# test_net ./logs/saved_scalers/TCN_20201018-003156/config.yaml # translation | value | all | complex



## All Slip Experiments
# train_epochs 200 ./logs/saved_scalers/TCN_20201031-133457/config.yaml # v1-3 |  0.05 &  0.5 | lr0.002 | [24, 24]
# test_net ./logs/saved_scalers/TCN_20201031-133457/config.yaml # v1-3 |  0.05 &  0.5 | lr0.002 | [24, 24]

## All direction Experiments
# train_epochs 200 ./logs/saved_scalers/TCN_20201031-133359/config.yaml # v1-3 |  0.05 &  0.5 | lr0.002 | [24, 24]
# test_net ./logs/saved_scalers/TCN_20201031-133359/config.yaml # v1-3 |  0.05 &  0.5 | lr0.002 | [24, 24]



####### NEW 3D TCN EXPERIMENTS #########
# Translation direction experiemnts
# train_epochs 100 ./logs/models/TCN3D_20201116-130917/config.yaml # v3 |  0.05 &  0.5 |
# train_epochs 50 ./logs/models/TCN3D_20201116-160240/config.yaml # v3 |  0.05 &  0.5 | 2+1D |
# train_epochs 50 ./logs/models/TCN3D_20201117-161519/config.yaml # v3 |  0.05 &  0.5 | 2+1D | with last row flipped
# train_epochs 50 ./logs/models/TCN3D_20201117-170351/config.yaml # v3 |  0.05 &  0.5 | with last row flipped

# test_net ./logs/models/TCN3D_20201117-161519/config.yaml # v3 |  0.05 &  0.5 | 2+1D | with last row flipped
# test_net ./logs/models/TCN3D_20201117-170351/config.yaml # v3 |  0.05 &  0.5 | with last row flipped
