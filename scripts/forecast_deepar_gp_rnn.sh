#!/bin/bash

python main.py \
--model_name DeepAR \
--dataset_name gp \
--mode 2 \
--condition_size 40 \
--horizon 20 \
--quantile 1 \
--sample_size 200 \
--cell_type GRU \
--dAR_hidden_size 128 \
--dAR_hidden_depth 3 \
--seed 200
