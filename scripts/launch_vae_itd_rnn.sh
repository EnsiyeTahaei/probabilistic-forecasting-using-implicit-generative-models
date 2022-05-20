#!/bin/bash

python main.py \
--model_name VAE \
--dataset_name itd \
--mode 1 \
--max_steps 50000 \
--patience 20000 \
--batch_size 128 \
--condition_size 16 \
--horizon 1 \
--optimizer_name Adam \
--lr 0.01 \
--cell_type GRU \
--latent_length 18 \
--dec_hidden_size 32 \
--dec_hidden_depth 2 \
--dec_dropout_rate 0.30 \
--enc_hidden_size 32 \
--enc_hidden_depth 3 \
--enc_dropout_rate 0.12 \
--hist_bins 80 \
--hist_min 7e8 \
--hist_max 9e9 \
--seed 200 \
--log_interval 5
