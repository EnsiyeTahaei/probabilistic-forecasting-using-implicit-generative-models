#!/bin/bash

python main.py \
--model_name AAE \
--dataset_name hepc \
--mode 1 \
--max_steps 50000 \
--ae_max_iter 1 \
--dis_max_iter 3 \
--patience 20000 \
--batch_size 128 \
--condition_size 31 \
--horizon 1 \
--optimizer_name Adam \
--lr 0.001 \
--cell_type TCN \
--latent_length 9 \
--dec_hidden_size 128 \
--dec_num_channel 118 \
--dec_dropout_rate 0.14 \
--enc_hidden_size 128 \
--enc_num_channel 98 \
--enc_dropout_rate 0.15 \
--num_layers 4 \
--kernel_size 2 \
--hist_bins 80 \
--hist_min 0 \
--hist_max 6 \
--seed 200 \
--log_interval 5
