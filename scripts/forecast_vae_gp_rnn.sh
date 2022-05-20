#!/bin/bash

python main.py \
--model_name VAE \
--dataset_name gp \
--mode 2 \
--condition_size 47 \
--latent_length 34 \
--horizon 20 \
--quantile 1 \
--sample_size 200 \
--cell_type GRU \
--dec_hidden_size 512 \
--dec_hidden_depth 2 \
--dec_dropout_rate 0. \
--enc_hidden_size 16 \
--enc_hidden_depth 3 \
--enc_dropout_rate 0.39 \
--seed 200
