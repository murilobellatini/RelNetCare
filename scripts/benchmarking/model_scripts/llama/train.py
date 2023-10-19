from src.paths import root_path
from src.paths import LOCAL_DATA_PATH

import os
import argparse
import subprocess
from dotenv import load_dotenv

load_dotenv()

# Create an argument parser
parser = argparse.ArgumentParser(description='Train LLaMA Lora model')

# Add arguments
parser.add_argument('--exp_group', type=str, default="BenchmarkTest", help='Experiment group')
parser.add_argument('--epoch_count', type=int, default=5, help='Number of epochs')
parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
parser.add_argument('--data_folder', type=str, default=f"{LOCAL_DATA_PATH}/processed/dialog-re-llama-11cls-rebalPairs-rwrtKeys-instrC-mxTrnCp3-skpTps", help='Data folder path')
parser.add_argument('--model_name', type=str, default='llama-7B-hf', help='Model name')
parser.add_argument('--merge_dev_train', type=bool, default=False, help='Model name')

# Parse the arguments
args = parser.parse_args()
data_stem = args.data_folder.split('/')[-1]


# Initialization
llama_lora_dir = f"{root_path}/llms-fine-tuning/llama-lora-fine-tuning"
hf_model_dir = f"{os.environ['MODEL_DIR']}/custom/{args.model_name}"
model_output_dir= f"{os.environ['MODEL_DIR']}/fine-tuned/{args.model_name}/{data_stem}"
lora_adaptor_name = f"{args.model_name}-lora-adaptor/{data_stem}"
lora_adaptor_dir = f"{os.environ['MODEL_DIR']}/custom/{lora_adaptor_name}"
data_path = f"{args.data_folder}/{data_stem}-train-dev.json" if args.merge_dev_train else f"{args.data_folder}/{data_stem}-train.json"

# Training
subprocess.run([
    "deepspeed", f"{llama_lora_dir}/fastchat/train/train_lora.py",
    "--deepspeed", f"{llama_lora_dir}/deepspeed-config.json",
    "--lora_r", "8",
    "--exp_group", args.exp_group,
    "--lora_alpha", "16",
    "--model_name_or_path", hf_model_dir,
    "--data_path", data_path,
    "--output_dir", lora_adaptor_dir,
    "--fp16", "True",
    "--num_train_epochs", str(args.epoch_count),
    "--per_device_train_batch_size", "4",
    "--per_device_eval_batch_size", "12",
    "--gradient_accumulation_steps", "1",
    "--save_strategy", "steps",
    "--save_steps", "1200",
    "--save_total_limit", "1",
    "--learning_rate", str(args.lr),  # Adjust this based on your Python environment variables
    "--weight_decay", "0.",
    "--warmup_ratio", "0.03",
    "--lr_scheduler_type", "cosine",
    "--logging_steps", "1",
    "--model_max_length", "1024",
    "--gradient_checkpointing", "True",
    "--lora_dropout", "0.05"
])

# Merge lora adaptor with llama for fine-tuned behavior
subprocess.run([
    "python", "-m", "fastchat.model.apply_lora",
    "--base", hf_model_dir,
    "--target", model_output_dir,
    "--lora", lora_adaptor_dir
])