import wandb
from datasets import load_metric
from transformers import BartForConditionalGeneration, BartTokenizer, EvalPrediction, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import json
import os
import torch

# Customizable settings
exp_group = "TryOutBART"  # Change this for each run
data_folder = "/home/murilo/RelNetCare/data/processed/dialog-re-llama-11cls-rebalPairs-rwrtKeys-instrC-mxTrnCp3-skpTps-prepBART"
freeze_encoder = True

# Initialize Weights and Biases with more args
args_dict = {
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "num_train_epochs": 3,
    'learning_rate': 5e-5,  # good starting point
    "exp_group": exp_group,
    "data_stem": data_folder.split('/')[-1],
    "data_folder": data_folder,
    "freeze_encoder": freeze_encoder,
    'truncation': True,
    'max_length': 1024,
    'model_name':'facebook/bart-base',
}
args_dict['output_dir'] = f"/mnt/vdb1/murilo/models/fine-tuned/{args_dict['model_name']}/{args_dict['data_stem']}"



metric = load_metric("rouge")

def compute_metrics(p: EvalPrediction):
    decoded_preds = [tokenizer.decode(pred, skip_special_tokens=True) for pred in p.predictions]
    decoded_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in p.label_ids]
    return metric.compute(predictions=decoded_preds, references=decoded_labels, rouge_types=["rougeL"])


# Initialize tokenizer and model
tokenizer = BartTokenizer.from_pretrained(args_dict['model_name'])
model = BartForConditionalGeneration.from_pretrained(args_dict['model_name'])

# Conditionally freeze encoder
if args_dict["freeze_encoder"]:
    for param in model.model.encoder.parameters():
        param.requires_grad = False

# Read and tokenize the datasets
def read_and_tokenize(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    inputs = [tokenizer(d['input'], padding='max_length', truncation=args_dict['truncation'], max_length=args_dict['max_length'], return_tensors='pt') for d in data]
    outputs = [tokenizer(d['output'], padding='max_length', truncation=args_dict['truncation'], max_length=args_dict['max_length'], return_tensors='pt') for d in data]
    return inputs, outputs

train_inputs, train_outputs = read_and_tokenize(os.path.join(data_folder, 'train.json'))
dev_inputs, dev_outputs = read_and_tokenize(os.path.join(data_folder, 'dev.json'))
test_inputs, test_outputs = read_and_tokenize(os.path.join(data_folder, 'test.json'))

print(f"Type of train_inputs: {type(train_inputs)}")
print(f"Type of dev_inputs: {type(dev_inputs)}")
print(f"Length of train_inputs: {len(train_inputs)}")
print(f"Length of dev_inputs: {len(dev_inputs)}")

# Tokenize and convert to numeric IDs
dummy_pred_ids = tokenizer("some prediction text", return_tensors="pt").input_ids
dummy_label_ids = tokenizer("some label text", return_tensors="pt").input_ids

# Use these token IDs for your dummy test
print("Dummy metrics:", compute_metrics(EvalPrediction(predictions=dummy_pred_ids, label_ids=dummy_label_ids)))

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Print device info
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# Move model to device
model.to(device)

# Initialize wandb
wandb.init(project="huggingface", config=args_dict)


# More Training args with wandb
training_args = Seq2SeqTrainingArguments(
    output_dir=args_dict['output_dir'],
    per_device_train_batch_size=args_dict['per_device_train_batch_size'],
    per_device_eval_batch_size=args_dict['per_device_eval_batch_size'],
    num_train_epochs=args_dict['num_train_epochs'],
    learning_rate=args_dict['learning_rate'],  # Adding learning rate
    logging_dir='./logs',
    logging_steps=100,
    save_steps=1_000,
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="wandb",
    push_to_hub=False,
    evaluation_strategy='steps',  # Add this line
    eval_steps=1_000,  # And this one
    load_best_model_at_end=True,
)

print(f"save_steps: {training_args.save_steps}")
print(f"eval_steps: {training_args.eval_steps}")

# Initialize the trainer with the model on the device
print("Initializing trainer...")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_inputs,
    eval_dataset=dev_inputs,
    compute_metrics=compute_metrics,
)

print("About to start training...")
print(f"Train dataset size: {len(train_inputs)}")
print(f"Dev dataset size: {len(dev_inputs)}")

# Fine-tuning
print("Training...")
trainer.train()
print("Training complete.")
