#!/bin/bash

# An array of learning rates you want to experiment with
learning_rates=(1e-4 5e-5 3e-5 1e-5 5e-6 1e-6)

# Number of epochs
epochs=5

for learning_rate in ${learning_rates[@]}; do

    # Convert learning rate to a string that can be safely used in a file name
    learning_rate_str=$(echo $learning_rate | tr . p | tr - m)

    python /mnt/vdb1/Development/murilo/RelNetCare/src/custom_dialogre/run_classifier.py \
        --task_name bert \
        --do_train \
        --do_eval \
        --data_dir /mnt/vdb1/Development/murilo/RelNetCare/data/processed/dialog-re-ternary-undersampled \
        --vocab_file /mnt/vdb1/Development/murilo/RelNetCare/models/downloaded/bert-base/vocab.txt \
        --bert_config_file /mnt/vdb1/Development/murilo/RelNetCare/models/downloaded/bert-base/bert_config.json \
        --init_checkpoint /mnt/vdb1/Development/murilo/RelNetCare/models/downloaded/bert-base/pytorch_model.bin \
        --max_seq_length 512 \
        --train_batch_size 24 \
        --learning_rate $learning_rate \
        --num_train_epochs $epochs \
        --output_dir /mnt/vdb1/Development/murilo/RelNetCare/models/fine-tuned/bert-base-dialog-re-ternary-undersampled/lr-${learning_rate_str} \
        --gradient_accumulation_steps 2 \
        --exp_group 005-DialogRETernaryUndersampled-BERT-Base \
        --relation_type_count 3
done
