#!/bin/bash

python main.py \
--model_name GAN \
--dataset_name itd \
--mode 1 \
--max_steps 50000 \
--dis_max_iter 5 \
--patience 10000 \
--batch_size 128 \
--condition_size 29 \
--horizon 1 \
--optimizer_name Adam \
--lr 0.01 \
--cell_type LSTM \
--noise_dim 19 \
--gen_hidden_size 128 \
--gen_hidden_depth 1 \
--dis_hidden_size 128 \
--dis_hidden_depth 1 \
--hist_bins 50 \
--hist_min 7e8 \
--hist_max 9e9 \
--seed 200 \
--log_interval 10
