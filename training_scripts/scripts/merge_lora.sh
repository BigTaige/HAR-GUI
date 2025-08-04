#!/bin/bash

#MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_NAME="./models/HAR-GUI-3B"

export PYTHONPATH=src:$PYTHONPATH

python src/merge_lora_weights.py \
    --model-path ../output/testing_lora \
    --model-base $MODEL_NAME  \
    --save-model-path ../output/merge_test \
    --safe-serialization