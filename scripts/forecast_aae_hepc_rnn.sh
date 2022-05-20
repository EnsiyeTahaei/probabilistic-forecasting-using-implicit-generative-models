#!/bin/bash

python main.py \
--model_name AAE \
--dataset_name hepc \
--mode 2 \
--condition_size 35 \
--latent_length 8 \
--horizon 20 \
--quantile 1 \
--sample_size 200 \
--cell_type GRU \
--dec_hidden_size 16 \
--dec_hidden_depth 2 \
--dec_dropout_rate 0.2 \
--enc_hidden_size 8 \
--enc_hidden_depth 2 \
--enc_dropout_rate 0.42 \
--seed 200
