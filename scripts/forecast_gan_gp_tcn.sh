#!/bin/bash

python main.py \
--model_name GAN \
--dataset_name gp \
--mode 2 \
--condition_size 29 \
--noise_dim 39 \
--horizon 20 \
--quantile 1 \
--sample_size 200 \
--cell_type TCN \
--gen_hidden_size 8 \
--gen_num_channel 78 \
--gen_dropout_rate 0. \
--dis_hidden_size 512 \
--dis_num_channel 36 \
--dis_dropout_rate 0. \
--num_layers 3 \
--kernel_size 3 \
--seed 200
