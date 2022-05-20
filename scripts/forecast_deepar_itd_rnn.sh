#!/bin/bash

python main.py \
--model_name DeepAR \
--dataset_name itd \
--mode 2 \
--condition_size 26 \
--horizon 20 \
--quantile 1 \
--sample_size 200 \
--cell_type GRU \
--dAR_hidden_size 16 \
--dAR_hidden_depth 2 \
--seed 200
