#!/bin/bash

# ===== Initialization =====

# Capture the command-line argument (overwrite or not)
MODE=$1  

# Dynamic variables based on certain conditions (e.g., model size)
model_size="7B"
dataset_name="dialog-re-llama-typed-pp-11cls-train-dev"
data_layer="processed/dialog-re-llama-typed-pp"

# Base paths
ROOT_DIR="/home/murilo/RelNetCare"
LLAMA_LORA_DIR="$ROOT_DIR/llms-fine-tuning/llama-lora-fine-tuning"
MODEL_DIR="$ROOT_DIR/models"
DATA_DIR="$ROOT_DIR/data/$data_layer"
CUSTOM_MODEL_DIR="$MODEL_DIR/custom"

# Derived paths
model_name="llama-$model_size-hf"
hf_model_dir="$CUSTOM_MODEL_DIR/$model_name"
data_path="$DATA_DIR/$dataset_name.json"
lora_adaptor_name="$model_name-lora-adaptor/$dataset_name"
lora_adaptor_dir="$CUSTOM_MODEL_DIR/$lora_adaptor_name"

# ===== Checkpoint Handling =====

if [ "$MODE" == "overwrite" ]; then
    echo "Overwrite mode on. Training will replace trained adapter in '$lora_adaptor_dir'."
    RESUME_CMD=""
else
    # Find the latest checkpoint by sorting the directory names and picking the last one
    LATEST_CHECKPOINT=$(ls $lora_adaptor_dir | grep 'global_step' | sort -n -t_ -k3 | tail -n1)
    
    if [ ! -z "$LATEST_CHECKPOINT" ]; then
        echo "Model checkpoint '$LATEST_CHECKPOINT' found. Resuming training..."
        RESUME_CMD="--resume_from_checkpoint $lora_adaptor_dir/$LATEST_CHECKPOINT"
    else
        echo "No checkpoints found. Training from scratch."
        RESUME_CMD=""
    fi
fi

# ===== Training =====

deepspeed "$LLAMA_LORA_DIR/fastchat/train/train_lora.py" \
    --deepspeed "$LLAMA_LORA_DIR/deepspeed-config.json" \
    $RESUME_CMD \
    --lora_r 8 \
    --lora_alpha 16 \
    --model_name_or_path "$hf_model_dir" \
    --data_path "$data_path" \
    --output_dir "$lora_adaptor_dir" \
    --fp16 True \
    --num_train_epochs 175 \
    --per_device_train_batch_size 12 \
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