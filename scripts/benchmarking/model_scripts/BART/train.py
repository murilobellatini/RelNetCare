import argparse
from transformers import AutoTokenizer
import torch
import wandb
from tqdm import tqdm  # <--- Use notebook version for Jupyter
from datasets import Dataset, DatasetDict
import os
import json
import evaluate
import numpy as np
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, BartConfig
from src.paths import LOCAL_DATA_PATH

from dotenv import load_dotenv
load_dotenv()

parser = argparse.ArgumentParser(description='Train BART Model')
parser.add_argument('--exp_group', default="BenchmarkTest", type=str, help='Experiment group')
parser.add_argument('--data_folder', type=str, default=f"{LOCAL_DATA_PATH}/processed/dialog-re-llama-11cls-rebalPairs-rwrtKeys-instrC-mxTrnCp3-skpTps-prepBART", help='Data folder path')
parser.add_argument('--model_name', type=str, default='facebook/bart-large', help='Model name')
parser.add_argument('--num_train_epochs', type=int, default=5, help='Model name')
parser.add_argument('--freeze_encoder', type=bool, default=False, help='Freeze encoder or not')
# add more arguments here
args = parser.parse_args()

# use args in the script like so:
exp_group = args.exp_group
data_folder = args.data_folder
freeze_encoder = args.freeze_encoder
model_name = args.model_name
num_train_epochs = args.num_train_epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_name == 'facebook/bart-base':
    train_batch_size = 24 # was 32
    eval_batch_size = 48 # was 64
elif model_name == 'facebook/bart-large':
    train_batch_size = 6
    eval_batch_size = 12
else:
    raise Exception("Model without batch size match")

# Initialize Weights and Biases with more args
args_dict = {
    "per_device_train_batch_size": train_batch_size,
    "per_device_eval_batch_size": eval_batch_size,
    "num_train_epochs": num_train_epochs,
    'learning_rate': 2e-5,  # good starting point: 2e-5 (min acceptable for large 2.5e-7)
    "exp_group": exp_group,
    "data_stem": data_folder.split('/')[-1],
    "data_folder": data_folder,
    "freeze_encoder": freeze_encoder,
    'truncation': True,
    'max_length': 512,
    'model_name':model_name,
    'memorization_task': False,
    'fp16': False,
    'merge_train_dev': False,
    'dropout_regularization_proba': None, # default 0.10
}
args_dict['output_dir'] = f"/mnt/vdb1/murilo/models/fine-tuned/{args_dict['model_name']}/{args_dict['data_stem']}"


checkpoint = args_dict['model_name']
tokenizer = AutoTokenizer.from_pretrained(args_dict['model_name'])


data_cap = -1
set_data = None
dataset_sets = {}
dict_sets = {}
for set_ in ('train', 'test', 'dev'):

    data_path = os.path.join(data_folder, f'{set_}.json')

    with open(data_path, 'r') as f:
        data = json.load(f)
            
    # Remap keys and separate into train/test
    if args_dict['memorization_task']:
        if not set_data:
            set_data = [{"text": item["input"], "summary": item["output"], "title": ""} for item in data[data_cap:]]
    else:
        set_data = [{"text": item["input"], "summary": item["output"], "title": ""} for item in data]
        
    # Merge 'train' and 'dev' if the flag is set
    if args_dict['merge_train_dev']:
        if set_ == 'dev':
            dict_sets['train'] = dict_sets['train'] + set_data
        else:
           dict_sets[set_] = set_data
    else:
        dict_sets[set_] = set_data
        

    
for set_ in ('train', 'test', 'dev'):
    if args_dict['merge_train_dev']:
        if set_ == 'dev':
            continue
    set_data = dict_sets[set_]
    dataset_sets[set_] = Dataset.from_dict(
        {"text": [item["text"] for item in set_data],
         "summary": [item["summary"] for item in set_data],
         "title": [item["title"] for item in set_data]}
        )
    

# Create DatasetDict
dataset_dict = DatasetDict(dataset_sets)
dataset_dict

def preprocess_function(examples, model_max_length=args_dict['max_length'], tokenizer_max_length=args_dict['max_length']):
    inputs = [doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=model_max_length, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=model_max_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_data = dataset_dict.map(preprocess_function, batched=True)




data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=args_dict['model_name'])


rouge = evaluate.load("rouge")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


try:
    # Release GPU
    model.cpu()
    del model
    torch.cuda.empty_cache()
except:
    print("No model loaded...")

if args_dict['dropout_regularization_proba']:
    config_extreme = BartConfig.from_pretrained(checkpoint,
                                                # encoder_layerdrop=0.2,
                                                # decoder_layerdrop=0.5,
                                                dropout=args_dict['dropout_regularization_proba'],
                                                # attention_dropout=0.5,
                                                # activation_dropout=0.5
                                                )

    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, config=config_extreme)

else:
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)


if args_dict['freeze_encoder']:
    print('freezing encoder!')
    for param in model.model.encoder.parameters():
        param.requires_grad = False


print("model.config.encoder_layerdrop=",model.config.encoder_layerdrop)
print("model.config.decoder_layerdrop=",model.config.decoder_layerdrop)
print("model.config.dropout=",model.config.dropout)
print("model.config.attention_dropout=",model.config.attention_dropout)
print("model.config.activation_dropout=",model.config.activation_dropout)

model.to(device)
args_dict



wandb.init(project="huggingface", config=args_dict, reinit=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=args_dict['output_dir'],
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=1,
    learning_rate=args_dict['learning_rate'],
    per_device_train_batch_size=args_dict['per_device_train_batch_size'],
    per_device_eval_batch_size=args_dict['per_device_eval_batch_size'],
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=args_dict['num_train_epochs'],
    predict_with_generate=True,
    fp16=args_dict['fp16'],
    load_best_model_at_end=True,
    # max_grad_norm=0.1,
    seed=42,
    # push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["dev"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
# @todo: set random seed fo training
trainer.train()

model.cpu()
del model
torch.cuda.empty_cache()

# with open("model_path.txt", "w") as f:
#     f.write(args_dict['output_dir'])
