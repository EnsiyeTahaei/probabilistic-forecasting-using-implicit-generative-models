#!/bin/bash

python main.py \
--model_name VAE \
--dataset_name hepc \
--mode 1 \
--max_steps 50000 \
--patience 20000 \
--batch_size 128 \
--condition_size 31 \
--horizon 1 \
--optimizer_name Adam \
--lr 0.001 \
--cell_type TCN \
--latent_length 31 \
--dec_hidden_size 8 \
--dec_num_channel 108 \
--dec_dropout_rate 0. \
--enc_hidden_size 64 \
--enc_num_channel 64 \
--enc_dropout_rate 0.26 \
--num_layers 4 \
--kernel_size 2 \
--hist_bins 80 \
--hist_min 0 \
--hist_max 6 \
--seed 200 \
--log_interval 5
