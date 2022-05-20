#!/bin/bash

python main.py \
--model_name VAE \
--dataset_name gp \
--mode 1 \
--max_steps 50000 \
--patience 20000 \
--batch_size 128 \
--condition_size 47 \
--horizon 1 \
--optimizer_name Adam \
--lr 0.001 \
--cell_type GRU \
--latent_length 34 \
--dec_hidden_size 512 \
--dec_hidden_depth 2 \
--dec_dropout_rate 0. \
--enc_hidden_size 16 \
--enc_hidden_depth 3 \
--enc_dropout_rate 0.39 \
--hist_bins 80 \
--hist_min 1e3 \
--hist_max 2e3 \
--seed 200 \
--log_interval 5
