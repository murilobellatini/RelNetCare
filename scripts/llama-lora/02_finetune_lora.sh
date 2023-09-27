#!/bin/bash

# Load variables from .env file
source /home/murilo/RelNetCare/.env

# ===== Initialization =====
mode=$1
llama_lora_dir="$ROOT_DIR/llms-fine-tuning/llama-lora-fine-tuning"
custom_model_dir="$MODEL_DIR/custom"
model_name="llama-$model_size-hf"
hf_model_dir="$custom_model_dir/$model_name"
datasets=("$data_stem") 

# Display variables
echo "=== Run Settings ==="
printf "Mode:\t\t"
if [ -z "$mode" ]; then
  echo "Not set (attempting to auto-resume)"
else
  echo "$mode"
fi
printf "Use Dev Set:\t$use_dev\n"
printf "Datasets:\t$datasets\n"
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

# Initialize run counter
run_counter=0

# Placeholder for latest trained model
latest_model="$hf_model_dir"

# Loop through datasets
for data_stem in "${datasets[@]}"; do
  let "run_counter++"

  data_layer="processed/$data_stem"
  DATA_DIR="$ROOT_DIR/data/$data_layer"

  if [ "$use_dev" == "true" ]; then
      echo "Using dev set...."
      dataset_name="$data_stem-train"
      eval_dataset_name="$data_stem-dev"
      EVAL_CMD=(--eval_data_path "$DATA_DIR/$eval_dataset_name.json" --evaluation_strategy "epoch" )
    #   EVAL_CMD=(--eval_data_path "$DATA_DIR/$eval_dataset_name.json" --evaluation_strategy "steps"  --eval_steps=200 )
  else
      echo "Using train-dev set...."
      dataset_name="$data_stem-train-dev"
      EVAL_CMD=(--evaluation_strategy "no")
  fi
  
  data_path="$DATA_DIR/$dataset_name.json"
  
  if [ "$lr" != "2e-5" ]; then
      lora_adaptor_name="${model_name}-lora-adaptor/${dataset_name}-${epoch_count}ep-${lr}lr"
  else
      lora_adaptor_name="${model_name}-lora-adaptor/${dataset_name}-${epoch_count}ep"
  fi
  
  lora_adaptor_dir="$custom_model_dir/$lora_adaptor_name/Run_$run_counter"

    # ===== Checkpoint Handling =====

    if [ "$mode" == "overwrite" ]; then
        echo "Overwrite mode on. Training will replace trained adapter in '$lora_adaptor_dir'."
        RESUME_CMD=""
    else
        # Find the latest checkpoint by sorting the directory names and picking the last one
        LATEST_CHECKPOINT=$(find $lora_adaptor_dir -type d -name 'checkpoint-*' | sort -V | tail -n1)

        if [ ! -z "$LATEST_CHECKPOINT" ]; then
            echo "Model checkpoint '$LATEST_CHECKPOINT' found. Resuming training..."
            RESUME_CMD="--resume_from_checkpoint $LATEST_CHECKPOINT"
        else
            echo "No checkpoints found. Training from scratch."
            RESUME_CMD=""
        fi
    fi

# ===== Training =====

  # Training command, using $latest_model as input
  deepspeed "$llama_lora_dir/fastchat/train/train_lora.py" \
        --deepspeed "$llama_lora_dir/deepspeed-config.json" \
        $RESUME_CMD \
        --lora_r 8 \
        --exp_group "$exp_group" \
        --lora_alpha 16 \
        --model_name_or_path "$latest_model" \
        --data_path "$data_path" \
        --output_dir "$lora_adaptor_dir" \
        --fp16 True \
        --num_train_epochs $epoch_count \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 12 \
        --gradient_accumulation_steps 1 \
        "${EVAL_CMD[@]}" \
        --save_strategy "steps" \
        --save_steps 1200 \
        --save_total_limit 1 \
        --learning_rate "$lr" \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --model_max_length 1024 \
        --gradient_checkpointing True \
        --lora_dropout 0.05

  # Update $latest_model for next iteration
  latest_model=$lora_adaptor_dir
  
  # Save parameters to README.md
  echo "Run_$run_counter" >> "$custom_model_dir/$lora_adaptor_name/RUNS_README.md"
  echo "Data Stem: $data_stem" >> "$custom_model_dir/$lora_adaptor_name/RUNS_README.md"
  echo "---" >> "$custom_model_dir/$lora_adaptor_name/RUNS_README.md"
done

# Copies the lates model to the model root folder
cp -R "$lora_adaptor_dir/"* "$custom_model_dir/$lora_adaptor_name/"
