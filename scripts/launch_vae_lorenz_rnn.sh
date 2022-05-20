#!/bin/bash

python main.py \
--model_name VAE \
--dataset_name lorenz \
--mode 1 \
--max_steps 50000 \
--patience 20000 \
--batch_size 128 \
--condition_size 24 \
--horizon 1 \
--optimizer_name Adam \
--lr 0.01 \
--cell_type GRU \
--latent_length 13 \
--dec_hidden_size 128 \
--dec_hidden_depth 3 \
--dec_dropout_rate 0.41 \
--enc_hidden_size 8 \
--enc_hidden_depth 3 \
--enc_dropout_rate 0.22 \
--hist_bins 80 \
--hist_min -11 \
--hist_max 11 \
--seed 200 \
--log_interval 5
