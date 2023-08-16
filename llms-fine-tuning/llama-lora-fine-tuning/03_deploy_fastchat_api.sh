#!/bin/bash

# Ask the user for input
echo "Which server do you want to start? (1 for gradio, 2 for open_ai): "
read choice

# Dynamic variables based on certain conditions (e.g., model size)
model_size="7B"
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

# Deploy FastChat API Wrapper
python3 -m fastchat.serve.controller &
python3 -m fastchat.serve.model_worker --model-name 'vicuna-7b-v1.1' --model-path "$output_dir"  &

# Introduce a delay to ensure background services have time to start
sleep 15  

# Check user's choice and start the desired server
if [ "$choice" == "1" ]; then
    # Start the gradio web server
    python3 -m fastchat.serve.gradio_web_server
elif [ "$choice" == "2" ]; then
    # Start the open_ai server
    python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
else
    echo "Invalid choice!"
fi

# Wait for all background processes to finish
wait
