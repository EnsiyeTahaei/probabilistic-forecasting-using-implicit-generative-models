#!/bin/bash

python main.py \
--model_name GAN \
--dataset_name itd \
--mode 2 \
--condition_size 19 \
--noise_dim 35 \
--horizon 20 \
--quantile 1 \
--sample_size 200 \
--cell_type TCN \
--gen_hidden_size 16 \
--gen_num_channel 120 \
--gen_dropout_rate 0. \
--dis_hidden_size 128 \
--dis_num_channel 115 \
--dis_dropout_rate 0. \
--num_layers 1 \
--kernel_size 10 \
--seed 200
