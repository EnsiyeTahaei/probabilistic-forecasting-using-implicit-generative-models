#!/bin/bash

python main.py \
--model_name VAE \
--dataset_name hepc \
--mode 1 \
--max_steps 50000 \
--patience 20000 \
--batch_size 128 \
--condition_size 28 \
--horizon 1 \
--optimizer_name RMSprop \
--lr 0.001 \
--cell_type GRU \
--latent_length 26 \
--dec_hidden_size 128 \
--dec_hidden_depth 2 \
--dec_dropout_rate 0.44 \
--enc_hidden_size 512 \
--enc_hidden_depth 3 \
--enc_dropout_rate 0. \
--hist_bins 80 \
--hist_min 0 \
--hist_max 6 \
--seed 200 \
--log_interval 5
