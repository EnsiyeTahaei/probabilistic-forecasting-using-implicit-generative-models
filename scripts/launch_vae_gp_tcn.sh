#!/bin/bash

python main.py \
--model_name VAE \
--dataset_name gp \
--mode 1 \
--max_steps 50000 \
--patience 20000 \
--batch_size 128 \
--condition_size 15 \
--horizon 1 \
--optimizer_name Adam \
--lr 0.01 \
--cell_type TCN \
--latent_length 44 \
--dec_hidden_size 128 \
--dec_num_channel 25 \
--dec_dropout_rate 0.27 \
--enc_hidden_size 512 \
--enc_num_channel 112 \
--enc_dropout_rate 0.19 \
--num_layers 3 \
--kernel_size 2 \
--hist_bins 80 \
--hist_min 1e3 \
--hist_max 2e3 \
--seed 200 \
--log_interval 5
