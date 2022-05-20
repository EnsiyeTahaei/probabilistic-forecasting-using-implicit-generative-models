#!/bin/bash

python main.py \
--model_name VAE \
--dataset_name itd \
--mode 2 \
--condition_size 16 \
--latent_length 18 \
--horizon 20 \
--quantile 1 \
--sample_size 200 \
--cell_type GRU \
--dec_hidden_size 32 \
--dec_hidden_depth 2 \
--dec_dropout_rate 0.30 \
--enc_hidden_size 32 \
--enc_hidden_depth 3 \
--enc_dropout_rate 0.12 \
--seed 200
