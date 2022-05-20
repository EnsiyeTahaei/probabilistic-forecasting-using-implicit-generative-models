#!/bin/bash

python main.py \
--model_name AAE \
--dataset_name gp \
--mode 1 \
--max_steps 50000 \
--ae_max_iter 1 \
--dis_max_iter 2 \
--patience 20000 \
--batch_size 128 \
--condition_size 11 \
--horizon 1 \
--optimizer_name RMSprop \
--lr 0.001 \
--cell_type LSTM \
--latent_length 26 \
--dec_hidden_size 128 \
--dec_hidden_depth 2 \
--dec_dropout_rate 0.25 \
--enc_hidden_size 32 \
--enc_hidden_depth 3 \
--enc_dropout_rate 0.31 \
--hist_bins 80 \
--hist_min 1e3 \
--hist_max 2e3 \
--seed 200 \
--log_interval 5
