#!/bin/bash

python main.py \
--model_name VAE \
--dataset_name itd \
--mode 1 \
--max_steps 50000 \
--patience 20000 \
--batch_size 128 \
--condition_size 27 \
--horizon 1 \
--optimizer_name Adam \
--lr 0.01 \
--cell_type TCN \
--latent_length 29 \
--dec_hidden_size 32 \
--dec_num_channel 63 \
--dec_dropout_rate 0.18 \
--enc_hidden_size 16 \
--enc_num_channel 5 \
--enc_dropout_rate 0.36 \
--num_layers 1 \
--kernel_size 14 \
--hist_bins 80 \
--hist_min 7e8 \
--hist_max 9e9 \
--seed 200 \
--log_interval 5
