#!/bin/bash

echo "Press Enter to start benchmark orchestration..."
read

# python /home/murilo/RelNetCare/scripts/benchmarking/model_scripts/BART/train.py \
#     --data_folder "/home/murilo/RelNetCare/data/processed/dialog-re-llama-11cls-rebalPairs-rwrtKeys-instrC-mxTrnCp3-skpTps-prepBART"

# python /home/murilo/RelNetCare/scripts/benchmarking/model_scripts/BART/infer_eval.py \
#     --data_folder "/home/murilo/RelNetCare/data/processed/dialog-re-llama-11cls-rebalPairs-rwrtKeys-instrC-mxTrnCp3-skpTps-prepBART"

python /home/murilo/RelNetCare/scripts/benchmarking/model_scripts/RebelBaseline/infer_eval.py \
    --data_folder "/home/murilo/RelNetCare/data/processed/dialog-re-llama-11cls-rebalPairs-rwrtKeys-instrC-mxTrnCp3-skpTps-prepBART"
