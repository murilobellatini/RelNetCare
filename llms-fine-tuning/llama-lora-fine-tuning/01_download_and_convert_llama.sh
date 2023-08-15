#!/bin/bash
# Dynamic variables based on certain conditions (e.g., model size)
model_size="7B"

# Base paths
ROOT_DIR="/home/murilo/RelNetCare/"
MODEL_DIR="$ROOT_DIR/models"
CUSTOM_MODEL_DIR="$MODEL_DIR/custom"

# Construct the model directory path using the base and specific paths
model_name="llama-$model_size-hf"
hf_model_dir="$CUSTOM_MODEL_DIR/$model_name"

# Export GIT variables for detailed logs
export GIT_TRACE=1
export GIT_CURL_VERBOSE=1

# Install pyllama from the specified git repository using pip
pip3 install git+https://github.com/juncongmoo/pyllama -i https://pypi.mirrors.ustc.edu.cn/simple --trusted-host=pypi.mirrors.ustc.edu.cn

# Download the llama model of the specified size using the llama module
python -m llama.download --model_size "$model_size"

# Convert downloaded llama weights to HuggingFace format and save to the specified output directory
CUDA_VISIBLE_DEVICES=1 python ./convert_llama_weights_to_hf.py --input_dir ./pyllama_data --model_size "$model_size" --output_dir "$hf_model_dir"

# Clean up by removing the pyllama_data directory
rm -rf ./pyllama_data
