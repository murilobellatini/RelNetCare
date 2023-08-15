#!/bin/bash
# Dynamic variables based on certain conditions (e.g., model size)
model_size="7B"
ip_address="10.195.2.147"
dataset_name="dummy_en"

# Base paths
ROOT_DIR="/home/murilo/RelNetCare"
MODEL_DIR="$ROOT_DIR/models"
CUSTOM_MODEL_DIR="$MODEL_DIR/custom"
DATA_DIR="$ROOT_DIR/data/raw/lora"
FINE_TUNED_MODEL_DIR="$MODEL_DIR/fine-tuned"

# Construct the model directory path using the base and specific paths
model_name="llama-$model_size-hf"
hf_model_dir="$CUSTOM_MODEL_DIR/$model_name"
data_path="$DATA_DIR/$dataset_name.json"
lora_adaptor_name="$model_name-lora-adaptor/$dataset_name"
output_dir="$FINE_TUNED_MODEL_DIR/$lora_adaptor_name"

# Port forwarding in the background
# ssh -L 8000:localhost:8000 murilo@$10.195.2.147 &

# Optional: symbolic wait to ensure SSH connection is established
# sleep 5

# Deploy FastChat API Wrapper
python3 -m fastchat.serve.controller &
python3 -m fastchat.serve.model_worker --model-name 'vicuna-7b-v1.1' --model-path "$output_dir" &
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000

# Wait for all background processes to finish (including SSH)


# wait
