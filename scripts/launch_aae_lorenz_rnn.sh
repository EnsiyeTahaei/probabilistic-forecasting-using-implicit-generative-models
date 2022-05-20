#!/bin/bash

python main.py \
--model_name AAE \
--dataset_name lorenz \
--mode 1 \
--max_steps 100000 \
--ae_max_iter 2 \
--dis_max_iter 3 \
--patience 60000 \
--batch_size 128 \
--condition_size 24 \
--horizon 1 \
--optimizer_name RMSprop \
--lr 0.001 \
--cell_type LSTM \
--latent_length 9 \
--dec_hidden_size 64 \
--dec_hidden_depth 3 \
--dec_dropout_rate 0.4 \
--enc_hidden_size 32 \
--enc_hidden_depth 2 \
--enc_dropout_rate 0.38 \
--hist_bins 80 \
--hist_min -11 \
--hist_max 11 \
--seed 200 \
--log_interval 5
