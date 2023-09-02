#!/bin/bash
# Dynamic variables based on certain conditions (e.g., model size)
model_size="7B"
epoch_count=5 #then 10 and 20
data_stem="dialog-re-llama-35cls-clsTskOnl"
dataset_name="$data_stem-train-dev"
data_layer="processed/$data_stem"

# Base paths
ROOT_DIR="/home/murilo/RelNetCare"
MODEL_DIR="/mnt/vdb1/murilo/models"
DATA_DIR="$ROOT_DIR/data/$data_layer"
CUSTOM_MODEL_DIR="$MODEL_DIR/custom"
FINE_TUNED_MODEL_DIR="$MODEL_DIR/fine-tuned"

# Construct the model directory path using the base and specific paths
model_name="llama-$model_size-hf"
hf_model_dir="$CUSTOM_MODEL_DIR/$model_name"
lora_adaptor_name="${model_name}-lora-adaptor/${dataset_name}-${epoch_count}ep"
lora_adaptor_dir="$CUSTOM_MODEL_DIR/$lora_adaptor_name"
output_dir="$FINE_TUNED_MODEL_DIR/$lora_adaptor_name"

# Merge lora adaptor with llama for fine-tuned behavior
python -m fastchat.model.apply_lora \
    --base "$hf_model_dir" \
    --target "$output_dir" \
    --lora "$lora_adaptor_dir"
