#!/bin/bash

python main.py \
--model_name AAE \
--dataset_name hepc \
--mode 2 \
--condition_size 31 \
--latent_length 9 \
--horizon 20 \
--quantile 1 \
--sample_size 200 \
--cell_type TCN \
--dec_hidden_size 128 \
--dec_num_channel 118 \
--dec_dropout_rate 0.14 \
--enc_hidden_size 128 \
--enc_num_channel 98 \
--enc_dropout_rate 0.15 \
--num_layers 4 \
--kernel_size 2 \
--seed 200
