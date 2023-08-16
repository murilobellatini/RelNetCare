#!/bin/bash
# Dynamic variables based on certain conditions (e.g., model size)
model_size="7B"
dataset_name="dialog-re-llama-train-dev"

# Base paths
ROOT_DIR="/home/murilo/RelNetCare"
MODEL_DIR="$ROOT_DIR/models"
FINE_TUNED_MODEL_DIR="$MODEL_DIR/fine-tuned"

# Construct the model directory path using the base and specific paths
model_name="llama-$model_size-hf"
lora_adaptor_name="$model_name-lora-adaptor/$dataset_name"
output_dir="$FINE_TUNED_MODEL_DIR/$lora_adaptor_name"

cleanup() {
    echo "Cleaning up..."
    for pid in "${pids[@]}"; do
        kill -9 "$pid" 2>/dev/null
    done
}

trap cleanup EXIT SIGINT SIGTERM

# Declare the PID array
declare -a pids

# Ask the user for input
echo "Which server do you want to start? (1 for gradio, 2 for open_ai): "
read choice

# Deploy FastChat API Wrapper
python3 -m fastchat.serve.controller &
pids+=("$!")

python3 -m fastchat.serve.model_worker --model-name 'vicuna-7b-v1.1' --model-path "$output_dir"  &
pids+=("$!")

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
