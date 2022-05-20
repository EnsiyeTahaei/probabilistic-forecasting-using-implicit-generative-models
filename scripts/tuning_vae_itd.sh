#!/bin/bash

pip install optuna

python main.py \
--model_name VAE \
--dataset_name itd \
--mode 0 \
--max_steps 5000 \
--batch_size 128 \
--horizon 1 \
--tune_cell TCN \
--max_device 1 \
--process_per_device 4 \
--seed 200