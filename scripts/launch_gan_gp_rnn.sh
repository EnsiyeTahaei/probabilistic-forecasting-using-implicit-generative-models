#!/bin/bash

python main.py \
--model_name GAN \
--dataset_name gp \
--mode 1 \
--max_steps 50000 \
--dis_max_iter 3 \
--patience 10000 \
--batch_size 128 \
--condition_size 36 \
--horizon 1 \
--optimizer_name RMSprop \
--lr 0.001 \
--cell_type GRU \
--noise_dim 26 \
--gen_hidden_size 64 \
--gen_hidden_depth 2 \
--dis_hidden_size 64 \
--dis_hidden_depth 3 \
--hist_bins 80 \
--hist_min 1e3 \
--hist_max 2e3 \
--seed 200 \
--log_interval 10
