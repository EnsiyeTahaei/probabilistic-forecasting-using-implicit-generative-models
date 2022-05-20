#!/bin/bash

python main.py \
--model_name AAE \
--dataset_name itd \
--mode 1 \
--max_steps 50000 \
--ae_max_iter 2 \
--dis_max_iter 2 \
--patience 20000 \
--batch_size 128 \
--condition_size 8 \
--horizon 1 \
--optimizer_name RMSprop \
--lr 0.001 \
--cell_type GRU \
--latent_length 20 \
--dec_hidden_size 128 \
--dec_hidden_depth 2 \
--dec_dropout_rate 0. \
--enc_hidden_size 128 \
--enc_hidden_depth 2 \
--enc_dropout_rate 0.37 \
--hist_bins 50 \
--hist_min 7e8 \
--hist_max 9e9 \
--seed 200 \
--log_interval 5
