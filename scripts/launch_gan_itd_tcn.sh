#!/bin/bash

python main.py \
--model_name GAN \
--dataset_name itd \
--mode 1 \
--max_steps 50000 \
--dis_max_iter 3 \
--patience 10000 \
--batch_size 128 \
--condition_size 19 \
--horizon 1 \
--optimizer_name RMSprop \
--lr 0.001 \
--cell_type TCN \
--noise_dim 35 \
--gen_hidden_size 16 \
--gen_num_channel 120 \
--gen_dropout_rate 0. \
--dis_hidden_size 128 \
--dis_num_channel 115 \
--dis_dropout_rate 0. \
--num_layers 1 \
--kernel_size 10 \
--hist_bins 50 \
--hist_min 7e8 \
--hist_max 9e9 \
--seed 200 \
--log_interval 10
