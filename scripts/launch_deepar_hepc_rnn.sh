#!/bin/bash

python main.py \
--model_name DeepAR \
--dataset_name hepc \
--mode 1 \
--max_steps 50000 \
--patience 10000 \
--batch_size 128 \
--condition_size 24 \
--horizon 1 \
--optimizer_name Adam \
--lr 0.01 \
--cell_type GRU \
--dAR_hidden_size 64 \
--dAR_hidden_depth 2 \
--hist_bins 80 \
--hist_min 0 \
--hist_max 6 \
--seed 200 \
--log_interval 100
