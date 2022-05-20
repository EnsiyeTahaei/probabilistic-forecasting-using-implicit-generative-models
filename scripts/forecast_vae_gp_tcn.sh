#!/bin/bash

python main.py \
--model_name VAE \
--dataset_name gp \
--mode 2 \
--condition_size 15 \
--latent_length 44 \
--horizon 20 \
--quantile 1 \
--sample_size 200 \
--cell_type TCN \
--dec_hidden_size 128 \
--dec_num_channel 25 \
--dec_dropout_rate 0.27 \
--enc_hidden_size 512 \
--enc_num_channel 112 \
--enc_dropout_rate 0.19 \
--num_layers 3 \
--kernel_size 2 \
--seed 200
