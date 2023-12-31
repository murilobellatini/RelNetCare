#!/bin/bash
idx="R1"
bert="bert-base"
data_dir="processed/dialog-re-11cls"
# data_dir="raw/dialog-re"
# relation_type_count=11
# relation_type_count=11
# relation_type_count=11
exp_goal="DialogREFocusRelations"

# params to change
learning_rates=(3e-5) 
epochs=(20)
patience=3
train_batch_size=24
classifier_layers=1
weight_decay_rate=0.01
frozen_bert=False

echo relation_type_count=$relation_type_count

# Split the string by '-'
IFS='-' read -r -a array <<< "$bert"
# Iterate over the array and capitalize the first character of each word
for index in "${!array[@]}"
do
    array[index]=$(echo ${array[index]^})
done
# Join the array back into a string
bert_clean=$(IFS='-'; echo "${array[*]}")

data_dir_clean=${data_dir#*/}

# Function to convert kebab-case to CamelCase
to_camel_case() {
    IFS='-' read -r -a array <<< "$1"
    for index in "${!array[@]}"; do
        array[index]=$(echo "${array[index]^}")
    done
    echo "${array[*]}" | tr -d ' '
}

bert_clean=$(to_camel_case "$bert_clean")
data_dir_clean=$(to_camel_case "$data_dir_clean")


for epoch in ${epochs[@]}; do
    for learning_rate in ${learning_rates[@]}; do
        # Convert learning rate to a string that can be safely used in a file name
        learning_rate_str=$(echo $learning_rate | tr . p | tr - m)

        # Set bert_frozen_flag and exp_group_suffix based on the frozen_bert value
        if [ "$frozen_bert" = True ] ; then
            bert_frozen_flag="--freeze_bert"
            exp_group_suffix="Frozen"
        else
            bert_frozen_flag=""
            exp_group_suffix="Unfrozen"
        fi

        exp_group=${idx}-${exp_goal}-${bert_clean}-${data_dir_clean}-${exp_group_suffix}
        output_dir=/mnt/vdb1/murilo/models/fine-tuned/${bert}-${data_dir_clean}/${exp_group_suffix}/${train_batch_size}bs-${classifier_layers}cls-${learning_rate_str}lr-${epochs}ep
        echo exp_group=$exp_group
        echo output_dir=$output_dir

        # Check if output directory is empty
        if [ "$(ls -A $output_dir)" ]; then
            read -p "Warning: $output_dir is not empty. Do you want to delete its contents? (y/n): " choice
            if [ "$choice" == "y" ]; then
                rm -rf $output_dir/*
                echo "Contents deleted."
            else
                echo "Contents not deleted. Exiting..."
                exit 1
            fi
        fi


        echo 

        python /home/murilo/RelNetCare/dialogre/bert/run_classifier.py \
            --task_name bert \
            --do_train \
            --do_eval \
            --data_dir /home/murilo/RelNetCare/data/${data_dir} \
            --vocab_file /mnt/vdb1/murilo/models/downloaded/${bert}/vocab.txt \
            --bert_config_file /mnt/vdb1/murilo/models/downloaded/${bert}/bert_config.json \
            --init_checkpoint /mnt/vdb1/murilo/models/downloaded/${bert}/pytorch_model.bin \
            --max_seq_length 512 \
            --train_batch_size $train_batch_size \
            --learning_rate $learning_rate \
            --num_train_epochs $epoch \
            --output_dir $output_dir \
            --gradient_accumulation_steps 2 \
            --exp_group $exp_group 
    done
done

echo output_dir=$output_dir
