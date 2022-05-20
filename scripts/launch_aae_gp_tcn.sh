#!/bin/bash

python main.py \
--model_name AAE \
--dataset_name gp \
--mode 1 \
--max_steps 50000 \
--ae_max_iter 2 \
--dis_max_iter 2 \
--patience 20000 \
--batch_size 128 \
--condition_size 31 \
--horizon 1 \
--optimizer_name RMSprop \
--lr 0.001 \
--cell_type TCN \
--latent_length 36 \
--dec_hidden_size 32 \
--dec_num_channel 29 \
--dec_dropout_rate 0. \
--enc_hidden_size 64 \
--enc_num_channel 59 \
--enc_dropout_rate 0.15 \
--num_layers 4 \
--kernel_size 2 \
--hist_bins 80 \
--hist_min 1e3 \
--hist_max 2e3 \
--seed 200 \
--log_interval 5
