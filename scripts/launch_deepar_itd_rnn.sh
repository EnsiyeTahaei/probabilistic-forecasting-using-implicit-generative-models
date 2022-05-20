#!/bin/bash

pip install properscoring

python main.py \
--model_name DeepAR \
--dataset_name itd \
--mode 1 \
--max_steps 50000 \
--patience 10000 \
--batch_size 128 \
--condition_size 26 \
--horizon 1 \
--optimizer_name RMSprop \
--lr 0.01 \
--cell_type GRU \
--dAR_hidden_size 16 \
--dAR_hidden_depth 2 \
--hist_bins 80 \
--hist_min 7e8 \
--hist_max 9e9 \
--seed 200 \
--log_interval 5
