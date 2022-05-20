#!/bin/bash

python main.py \
--model_name AAE \
--dataset_name hepc \
--mode 1 \
--max_steps 50000 \
--ae_max_iter 3 \
--dis_max_iter 5 \
--patience 20000 \
--batch_size 128 \
--condition_size 35 \
--horizon 1 \
--optimizer_name Adam \
--lr 0.001 \
--cell_type GRU \
--latent_length 8 \
--dec_hidden_size 16 \
--dec_hidden_depth 2 \
--dec_dropout_rate 0.2 \
--enc_hidden_size 8 \
--enc_hidden_depth 2 \
--enc_dropout_rate 0.42 \
--hist_bins 80 \
--hist_min 0 \
--hist_max 6 \
--seed 200 \
--log_interval 5
