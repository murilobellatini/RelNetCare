#!/bin/bash

# Load variables from .env file
source /home/murilo/RelNetCare/.env

# Dynamic variables based on certain conditions (e.g., model size)
data_layer="processed/$data_stem"
data_dir="$ROOT_DIR/data/$data_layer"
custom_model_dir="$MODEL_DIR/custom"
fine_tuned_model_dir="$MODEL_DIR/fine-tuned"

# Display variables
echo "=== Run Settings ==="
printf "Use Dev Set:\t$use_dev\n"
printf "Dataset:\t$data_stem\n"
printf "Model Size:\t$model_size\n"
printf "Learning Rate:\t$lr\n"
printf "Epoch Count:\t$epoch_count\n"
printf "Exp. Group:\t$exp_group\n"
echo "===================="

# Ask for confirmation
read -p "Are you OK with these settings? (y/n) " answer

# Check answer
if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
  echo "Great! Running the script."
  # Your code here...
else
  echo "Exiting."
  exit 1
fi

if [ "$use_dev" == "true" ]; then
    dataset_name="$data_stem-train"
else
    dataset_name="$data_stem-train-dev"
fi

# Construct the model directory path using the base and specific paths
model_name="llama-$model_size-hf"
hf_model_dir="$custom_model_dir/$model_name"
if [ "$lr" != "2e-5" ]; then
lora_adaptor_name="${model_name}-lora-adaptor/${dataset_name}-${epoch_count}ep-${lr}lr"
else
lora_adaptor_name="${model_name}-lora-adaptor/${dataset_name}-${epoch_count}ep"
fi
lora_adaptor_dir="$custom_model_dir/$lora_adaptor_name"
output_dir="$fine_tuned_model_dir/$lora_adaptor_name"

# Merge lora adaptor with llama for fine-tuned behavior
python -m fastchat.model.apply_lora \
    --base "$hf_model_dir" \
    --target "$output_dir" \
    --lora "$lora_adaptor_dir"
