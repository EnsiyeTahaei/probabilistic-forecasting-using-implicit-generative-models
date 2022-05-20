#!/bin/bash

python main.py \
--model_name GAN \
--dataset_name hepc \
--mode 2 \
--condition_size 27 \
--noise_dim 20 \
--horizon 20 \
--quantile 1 \
--sample_size 200 \
--cell_type TCN \
--gen_hidden_size 32 \
--gen_num_channel 71 \
--gen_dropout_rate 0. \
--dis_hidden_size 16 \
--dis_num_channel 73 \
--dis_dropout_rate 0. \
--num_layers 1 \
--kernel_size 14 \
--seed 200
