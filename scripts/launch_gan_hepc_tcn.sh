#!/bin/bash

python main.py \
--model_name GAN \
--dataset_name hepc \
--mode 1 \
--max_steps 50000 \
--dis_max_iter 5 \
--patience 10000 \
--batch_size 128 \
--condition_size 27 \
--horizon 1 \
--optimizer_name Adam \
--lr 0.001 \
--cell_type TCN \
--noise_dim 20 \
--gen_hidden_size 32 \
--gen_num_channel 71 \
--gen_dropout_rate 0. \
--dis_hidden_size 16 \
--dis_num_channel 73 \
--dis_dropout_rate 0. \
--num_layers 1 \
--kernel_size 14 \
--hist_bins 80 \
--hist_min 0 \
--hist_max 6 \
--seed 200 \
--log_interval 5
