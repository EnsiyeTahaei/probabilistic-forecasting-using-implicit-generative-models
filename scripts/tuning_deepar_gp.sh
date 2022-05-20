#!/bin/bash

pip install optuna

python main.py \
--model_name DeepAR \
--dataset_name gp \
--mode 0 \
--max_steps 5000 \
--batch_size 128 \
--horizon 1 \
--tune_cell RNN \
--max_device 1 \
--process_per_device 3 \
--seed 200