#!/bin/bash

python main.py \
--model_name GAN \
--dataset_name hepc \
--mode 1 \
--max_steps 50000 \
--dis_max_iter 5 \
--patience 10000 \
--batch_size 128 \
--condition_size 37 \
--horizon 1 \
--optimizer_name RMSprop \
--lr 0.001 \
--cell_type LSTM \
--noise_dim 25 \
--gen_hidden_size 32 \
--gen_hidden_depth 2 \
--dis_hidden_size 64 \
--dis_hidden_depth 1 \
--hist_bins 80 \
--hist_min 0 \
--hist_max 6 \
--seed 200 \
--log_interval 10
