#!/bin/bash

python main.py \
--model_name VAE \
--dataset_name hepc \
--mode 2 \
--condition_size 31 \
--latent_length 31 \
--horizon 20 \
--quantile 1 \
--sample_size 200 \
--cell_type TCN \
--dec_hidden_size 8 \
--dec_num_channel 108 \
--dec_dropout_rate 0. \
--enc_hidden_size 64 \
--enc_num_channel 64 \
--enc_dropout_rate 0.26 \
--num_layers 4 \
--kernel_size 2 \
--seed 200
