#!/bin/bash

python main.py \
--model_name AAE \
--dataset_name gp \
--mode 2 \
--condition_size 31 \
--latent_length 36 \
--horizon 20 \
--quantile 1 \
--sample_size 200 \
--cell_type TCN \
--dec_hidden_size 32 \
--dec_num_channel 29 \
--dec_dropout_rate 0. \
--enc_hidden_size 64 \
--enc_num_channel 59 \
--enc_dropout_rate 0.15 \
--num_layers 4 \
--kernel_size 2 \
--seed 200
