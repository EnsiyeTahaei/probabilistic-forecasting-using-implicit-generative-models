#!/bin/bash

python main.py \
--model_name AAE \
--dataset_name gp \
--mode 2 \
--condition_size 11 \
--latent_length 26 \
--horizon 20 \
--quantile 1 \
--sample_size 200 \
--cell_type LSTM \
--dec_hidden_size 128 \
--dec_hidden_depth 2 \
--dec_dropout_rate 0.25 \
--enc_hidden_size 32 \
--enc_hidden_depth 3 \
--enc_dropout_rate 0.31 \
--seed 200
