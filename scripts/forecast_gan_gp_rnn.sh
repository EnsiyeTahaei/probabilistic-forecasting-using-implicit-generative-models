#!/bin/bash

python main.py \
--model_name GAN \
--dataset_name gp \
--mode 2 \
--condition_size 36 \
--noise_dim 26 \
--horizon 20 \
--quantile 1 \
--sample_size 200 \
--cell_type GRU \
--gen_hidden_size 64 \
--gen_hidden_depth 2 \
--dis_hidden_size 64 \
--dis_hidden_depth 3 \
--seed 200
