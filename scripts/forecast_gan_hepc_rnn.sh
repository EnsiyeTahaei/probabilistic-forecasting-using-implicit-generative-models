#!/bin/bash

python main.py \
--model_name GAN \
--dataset_name hepc \
--mode 2 \
--condition_size 37 \
--noise_dim 25 \
--horizon 20 \
--quantile 1 \
--sample_size 200 \
--cell_type LSTM \
--gen_hidden_size 32 \
--gen_hidden_depth 2 \
--dis_hidden_size 64 \
--dis_hidden_depth 1 \
--seed 200
