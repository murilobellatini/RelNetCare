#!/bin/bash

# Load variables from .env file
source /home/murilo/RelNetCare/.env

if [ "$use_dev" == "true" ]; then
    dataset_name="$data_stem-train"
else
    dataset_name="$data_stem-train-dev"
fi
# Base paths
fine_tuned_model_dir="$MODEL_DIR/fine-tuned"

# Construct the model directory path using the base and specific paths
model_name="llama-$model_size-hf"
if [ "$lr" != "2e-5" ]; then
lora_adaptor_name="${model_name}-lora-adaptor/${dataset_name}-${epoch_count}ep-${lr}lr"
else
lora_adaptor_name="${model_name}-lora-adaptor/${dataset_name}-${epoch_count}ep"
fi
output_dir="$fine_tuned_model_dir/$lora_adaptor_name"
echo "output_dir=$output_dir"

cleanup() {
    echo "Cleaning up..."
    for pid in "${pids[@]}"; do
        kill -9 "$pid" 2>/dev/null
    done
}

trap cleanup EXIT SIGINT SIGTERM

# Declare the PID array
declare -a pids

# Display variables
echo "=== Run Settings ==="
printf "Use Dev Set:\t$use_dev\n"
printf "Dataset:\t$data_stem\n"
printf "Model Size:\t$model_size\n"
printf "Learning Rate:\t$lr\n"
printf "Epoch Count:\t$epoch_count\n"
printf "Exp. Group:\t$exp_group\n"
echo "===================="

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
