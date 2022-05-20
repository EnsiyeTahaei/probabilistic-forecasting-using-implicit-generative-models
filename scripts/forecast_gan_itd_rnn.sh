#!/bin/bash

python main.py \
--model_name GAN \
--dataset_name itd \
--mode 2 \
--condition_size 29 \
--noise_dim 19 \
--horizon 20 \
--quantile 1 \
--sample_size 200 \
--cell_type LSTM \
--gen_hidden_size 128 \
--gen_hidden_depth 1 \
--dis_hidden_size 128 \
--dis_hidden_depth 1 \
--seed 200
