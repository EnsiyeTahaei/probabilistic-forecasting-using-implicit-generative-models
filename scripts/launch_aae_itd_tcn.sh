#!/bin/bash

python main.py \
--model_name AAE \
--dataset_name itd \
--mode 1 \
--max_steps 50000 \
--ae_max_iter 2 \
--dis_max_iter 5 \
--patience 20000 \
--batch_size 128 \
--condition_size 31 \
--horizon 1 \
--optimizer_name Adam \
--lr 0.001 \
--cell_type TCN \
--latent_length 25 \
--dec_hidden_size 128 \
--dec_num_channel 115 \
--dec_dropout_rate 0. \
--enc_hidden_size 16 \
--enc_num_channel 125 \
--enc_dropout_rate 0.3 \
--num_layers 4 \
--kernel_size 2 \
--hist_bins 80 \
--hist_min 7e8 \
--hist_max 9e9 \
--seed 200 \
--log_interval 5
