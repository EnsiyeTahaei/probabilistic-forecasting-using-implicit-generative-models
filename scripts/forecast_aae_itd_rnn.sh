#!/bin/bash

python main.py \
--model_name AAE \
--dataset_name itd \
--mode 2 \
--condition_size 8 \
--latent_length 20 \
--horizon 20 \
--quantile 1 \
--sample_size 200 \
--cell_type GRU \
--dec_hidden_size 128 \
--dec_hidden_depth 2 \
--dec_dropout_rate 0. \
--enc_hidden_size 128 \
--enc_hidden_depth 2 \
--enc_dropout_rate 0.37 \
--seed 200
