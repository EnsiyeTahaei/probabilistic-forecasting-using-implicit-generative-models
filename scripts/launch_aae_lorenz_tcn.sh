#!/bin/bash

python main.py \
--model_name AAE \
--dataset_name lorenz \
--mode 1 \
--max_steps 100000 \
--ae_max_iter 1 \
--dis_max_iter 3 \
--patience 80000 \
--batch_size 128 \
--condition_size 24 \
--horizon 1 \
--optimizer_name Adam \
--lr 0.001 \
--cell_type TCN \
--latent_length 10 \
--dec_hidden_size 512 \
--dec_num_channel 16 \
--dec_dropout_rate 0.48 \
--enc_hidden_size 64 \
--enc_num_channel 50 \
--enc_dropout_rate 0.25 \
--num_layers 2 \
--kernel_size 5 \
--hist_bins 80 \
--hist_min -11 \
--hist_max 11 \
--seed 200 \
--log_interval 5
