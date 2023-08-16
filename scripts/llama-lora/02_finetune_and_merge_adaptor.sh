#!/bin/bash
# Dynamic variables based on certain conditions (e.g., model size)
model_size="7B"
dataset_name="dialog-re-llama-train-dev"
data_layer="processed/dialog-re-llama"

# Base paths
ROOT_DIR="/home/murilo/RelNetCare"
LLAMA_LORA_DIR=$ROOT_DIR/llms-fine-tuning/llama-lora-fine-tuning
MODEL_DIR="$ROOT_DIR/models"
DATA_DIR="$ROOT_DIR/data/$data_layer"
CUSTOM_MODEL_DIR="$MODEL_DIR/custom"
FINE_TUNED_MODEL_DIR="$MODEL_DIR/fine-tuned"

# Construct the model directory path using the base and specific paths
model_name="llama-$model_size-hf"
hf_model_dir="$CUSTOM_MODEL_DIR/$model_name"
data_path="$DATA_DIR/$dataset_name.json"
lora_adaptor_name="$model_name-lora-adaptor/$dataset_name"
lora_adaptor_dir="$CUSTOM_MODEL_DIR/$lora_adaptor_name"
output_dir="$FINE_TUNED_MODEL_DIR/$lora_adaptor_name"

# Train lora (for fine-tuning llama)
deepspeed "$LLAMA_LORA_DIR/fastchat/train/train_lora.py" \
    --deepspeed "$LLAMA_LORA_DIR/deepspeed-config.json" \
    --lora_r 8 \
    --lora_alpha 16 \
    --model_name_or_path "$hf_model_dir" \
    --data_path "$data_path" \
    --output_dir "$lora_adaptor_dir" \
    --fp16 True \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1\
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --lora_dropout 0.05

# Merge lora adaptor with llama for fine-tuned behavior
python -m fastchat.model.apply_lora \
    --base "$hf_model_dir" \
    --target "$output_dir" \
    --lora "$lora_adaptor_dir"
