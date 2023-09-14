#!/bin/bash
# 
# @TODO: include LLaMA-2, more info here https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md
# 

# ===== Initialization =====
MODE=$1
model_size="7B"
lr="2e-5" # default: 2e-5 / best performing 1.325e-5
# exp_group="DialogREReproduceRelCls"
# exp_group="DialogRETripletToTxt"
exp_group="DialogREExtractTriplets_ImprvNullRel"
# exp_group="DialogREExtractTriplets_GrpRels" 
epoch_count=5
ROOT_DIR="/home/murilo/RelNetCare"
LLAMA_LORA_DIR="$ROOT_DIR/llms-fine-tuning/llama-lora-fine-tuning"
MODEL_DIR="/mnt/vdb1/murilo/models"
CUSTOM_MODEL_DIR="$MODEL_DIR/custom"
model_name="llama-$model_size-hf"
hf_model_dir="$CUSTOM_MODEL_DIR/$model_name"
use_dev=true

# List of datasets
datasets=("dialog-re-llama-11cls-rebalPairs4x-rwrtKeys-instrC-mxTrnCp3-shfflDt") 
# datasets=("dialog-re-llama-36cls-clsTskOnl-rebalPairs2.5x-instrB-shfflDt-WthNRltnUndrsmpld")

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
      dataset_name="$data_stem-train"
      eval_dataset_name="$data_stem-dev"
      EVAL_CMD=(--eval_data_path "$DATA_DIR/$eval_dataset_name.json" --evaluation_strategy "epoch" )
    #   EVAL_CMD=(--eval_data_path "$DATA_DIR/$eval_dataset_name.json" --evaluation_strategy "steps"  --eval_steps=200 )
  else
      dataset_name="$data_stem-train-dev"
      EVAL_CMD=(--evaluation_strategy "no")
  fi
  
  data_path="$DATA_DIR/$dataset_name.json"
  
  if [ "$lr" != "2e-5" ]; then
      lora_adaptor_name="${model_name}-lora-adaptor/${dataset_name}-${epoch_count}ep-${lr}lr"
  else
      lora_adaptor_name="${model_name}-lora-adaptor/${dataset_name}-${epoch_count}ep"
  fi
  
  lora_adaptor_dir="$CUSTOM_MODEL_DIR/$lora_adaptor_name/Run_$run_counter"

    # ===== Checkpoint Handling =====

    if [ "$MODE" == "overwrite" ]; then
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
  deepspeed "$LLAMA_LORA_DIR/fastchat/train/train_lora.py" \
        --deepspeed "$LLAMA_LORA_DIR/deepspeed-config.json" \
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
  echo "Run_$run_counter" >> "$CUSTOM_MODEL_DIR/$lora_adaptor_name/RUNS_README.md"
  echo "Data Stem: $data_stem" >> "$CUSTOM_MODEL_DIR/$lora_adaptor_name/RUNS_README.md"
  echo "---" >> "$CUSTOM_MODEL_DIR/$lora_adaptor_name/RUNS_README.md"
done

# Copies the lates model to the model root folder
cp -R "$lora_adaptor_dir/"* "$CUSTOM_MODEL_DIR/$lora_adaptor_name/"
