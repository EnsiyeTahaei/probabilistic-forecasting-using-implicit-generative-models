#!/bin/bash

python main.py \
--model_name DeepAR \
--dataset_name gp \
--mode 1 \
--max_steps 50000 \
--patience 10000 \
--batch_size 128 \
--condition_size 40 \
--horizon 1 \
--optimizer_name RMSprop \
--lr 0.001 \
--cell_type GRU \
--dAR_hidden_size 128 \
--dAR_hidden_depth 3 \
--hist_bins 80 \
--hist_min 1e3 \
--hist_max 2e3 \
--seed 200 \
--log_interval 5
