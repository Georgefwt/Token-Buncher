#!/bin/bash

save_dir="eval_results/"
model=<MODEL_PATH>
global_record_file="eval_results/<MODEL_NAME>/eval_record_collection.csv"
selected_subjects="all"
gpu_util=0.8

python evaluate_from_local.py \
                 --selected_subjects $selected_subjects \
                 --save_dir $save_dir \
                 --model $model \
                 --global_record_file $global_record_file \
                 --gpu_util $gpu_util \
                 --ntrain 2 \
                 --use_chat_template



