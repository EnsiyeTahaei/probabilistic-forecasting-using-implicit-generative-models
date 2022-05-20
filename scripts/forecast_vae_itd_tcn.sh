#!/bin/bash

python main.py \
--model_name VAE \
--dataset_name itd \
--mode 2 \
--condition_size 27 \
--latent_length 29 \
--horizon 20 \
--quantile 1 \
--sample_size 200 \
--cell_type TCN \
--dec_hidden_size 32 \
--dec_num_channel 63 \
--dec_dropout_rate 0.18 \
--enc_hidden_size 16 \
--enc_num_channel 5 \
--enc_dropout_rate 0.36 \
--num_layers 1 \
--kernel_size 14 \
--seed 200
