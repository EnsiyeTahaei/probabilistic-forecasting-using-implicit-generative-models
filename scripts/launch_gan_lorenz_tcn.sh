#!/bin/bash

python main.py \
--model_name GAN \
--dataset_name lorenz \
--mode 1 \
--max_steps 50000 \
--dis_max_iter 4 \
--patience 10000 \
--batch_size 128 \
--condition_size 24 \
--horizon 1 \
--optimizer_name RMSprop \
--lr 0.001 \
--cell_type TCN \
--noise_dim 40 \
--gen_hidden_size 16 \
--gen_num_channel 80 \
--gen_dropout_rate 0. \
--dis_hidden_size 64 \
--dis_num_channel 100 \
--dis_dropout_rate 0. \
--num_layers 2 \
--kernel_size 5 \
--hist_bins 80 \
--hist_min -11 \
--hist_max 11 \
--seed 200 \
--log_interval 10
