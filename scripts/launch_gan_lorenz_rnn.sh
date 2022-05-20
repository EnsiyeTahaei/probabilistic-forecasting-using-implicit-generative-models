#!/bin/bash

python main.py \
--model_name GAN \
--dataset_name lorenz \
--mode 1 \
--max_steps 50000 \
--dis_max_iter 5 \
--patience 10000 \
--batch_size 128 \
--condition_size 24 \
--horizon 1 \
--optimizer_name Adam \
--lr 0.01 \
--cell_type GRU \
--noise_dim 25 \
--gen_hidden_size 8 \
--gen_hidden_depth 3 \
--dis_hidden_size 32 \
--dis_hidden_depth 2 \
--hist_bins 80 \
--hist_min -11 \
--hist_max 11 \
--seed 200 \
--log_interval 10
