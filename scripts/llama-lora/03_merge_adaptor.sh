#!/bin/bash
# Dynamic variables based on certain conditions (e.g., model size)
model_size="7B"
dataset_name="dialog-re-llama-typed-pp-11cls-train-dev"
data_layer="processed/dialog-re-llama-typed-pp"

# Base paths
ROOT_DIR="/home/murilo/RelNetCare"
MODEL_DIR="$ROOT_DIR/models"
DATA_DIR="$ROOT_DIR/data/$data_layer"
CUSTOM_MODEL_DIR="$MODEL_DIR/custom"
FINE_TUNED_MODEL_DIR="$MODEL_DIR/fine-tuned"

# Construct the model directory path using the base and specific paths
model_name="llama-$model_size-hf"
hf_model_dir="$CUSTOM_MODEL_DIR/$model_name"
lora_adaptor_name="$model_name-lora-adaptor/$dataset_name"
lora_adaptor_dir="$CUSTOM_MODEL_DIR/$lora_adaptor_name"
output_dir="$FINE_TUNED_MODEL_DIR/$lora_adaptor_name"

# Merge lora adaptor with llama for fine-tuned behavior
python -m fastchat.model.apply_lora \
    --base "$hf_model_dir" \
    --target "$output_dir" \
    --lora "$lora_adaptor_dir"
