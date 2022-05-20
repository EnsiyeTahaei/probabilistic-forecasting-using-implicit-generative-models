#!/bin/bash

python main.py \
--model_name AAE \
--dataset_name itd \
--mode 2 \
--condition_size 31 \
--latent_length 25 \
--horizon 20 \
--quantile 1 \
--sample_size 200 \
--cell_type TCN \
--dec_hidden_size 128 \
--dec_num_channel 115 \
--dec_dropout_rate 0. \
--enc_hidden_size 16 \
--enc_num_channel 125 \
--enc_dropout_rate 0.3 \
--num_layers 4 \
--kernel_size 2 \
--seed 200
