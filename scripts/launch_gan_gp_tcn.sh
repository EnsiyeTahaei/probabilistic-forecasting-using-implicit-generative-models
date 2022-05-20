#!/bin/bash

python main.py \
--model_name GAN \
--dataset_name gp \
--mode 1 \
--max_steps 50000 \
--dis_max_iter 4 \
--patience 10000 \
--batch_size 128 \
--condition_size 29 \
--horizon 1 \
--optimizer_name RMSprop \
--lr 0.001 \
--cell_type TCN \
--noise_dim 39 \
--gen_hidden_size 8 \
--gen_num_channel 78 \
--gen_dropout_rate 0. \
--dis_hidden_size 512 \
--dis_num_channel 36 \
--dis_dropout_rate 0. \
--num_layers 3 \
--kernel_size 3 \
--hist_bins 80 \
--hist_min 1e3 \
--hist_max 2e3 \
--seed 200 \
--log_interval 10
