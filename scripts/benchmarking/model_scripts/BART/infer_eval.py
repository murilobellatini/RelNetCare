import argparse
from datasets import Dataset, DatasetDict
import os
import json
import torch
from src.processing.relation_extraction_evaluator import RelationExtractorEvaluator, GranularMetricVisualizer
from src.config import get_config_from_stem
from src.processing.bart_processing import RelationConverter
from nltk.tokenize import sent_tokenize
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
nltk.download('punkt')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Run inferences on data for BART Model evaluation')
parser.add_argument('--data_folder', type=str, default="/home/murilo/RelNetCare/data/processed/dialog-re-llama-11cls-rebalPairs-rwrtKeys-instrC-mxTrnCp3-skpTps-prepBART", help='Data folder path')
parser.add_argument('--model_name', type=str, default='facebook/bart-base', help='Model name')
# add more arguments here
args = parser.parse_args()

# use args in the script like so:
data_folder = args.data_folder
base_model = args.model_name
data_stem = data_folder.split('/')[-1]
model_path = f"/mnt/vdb1/murilo/models/fine-tuned/{base_model}/{data_stem}"

data_cap = -1
set_data = None
dataset_sets = {}
dict_sets = {}

args_dict = {
    'memorization_task': False,
    'merge_train_dev': False
}

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

# Specify your directory path
all_folders = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))]

# If there's only one folder, set it as the checkpoint folder
if len(all_folders) == 1:
    checkpoint_folder = all_folders[0]
    full_checkpoint_path = os.path.join(model_path, checkpoint_folder)
    print(full_checkpoint_path)
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(full_checkpoint_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(full_checkpoint_path).to(device)
else:
    raise Exception("More than one checkpoint found. Be more specific!")

# Initialize an empty list to store raw_predicted_labels
raw_predicted_labels = []

# Loop through the test set
for entry in tqdm(dataset_dict['test']):
    text = entry['text']
    inputs = tokenizer(text, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(inputs, max_length=300, do_sample=False)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw_predicted_labels.append(summary)

# Confirm the shape
assert len(raw_predicted_labels) == len(dataset_dict['test'])


raw_predicted_labels[:10]

def convert_raw_labels_to_relations(raw_predicted_labels, input_path=''):
    converter = RelationConverter(input_path=input_path)
    predicted_labels = []
    error_count = 0  # Initialize error counter
    errors = []
    for raw_label in tqdm(raw_predicted_labels):
        label_relations = []
        if raw_label:
            for sent in sent_tokenize(raw_label): 
                try:
                    sub, rel, obj = converter._convert_sentence_to_relation(sent[:-1] if sent.endswith('.') else sent)
                except Exception as e:
                    sub, rel, obj = ("ERROR", "ERROR", "ERROR")
                    error = f'Exception `{e}` with sentence below:\n`{sent}`'
                    errors.append(error)
                    error_count += 1  # Increment error counter
                label_relations.append({'subject': sub, 'relation': rel, 'object': obj})

        predicted_labels.append(label_relations)

    # Confirm the shape
    assert len(raw_predicted_labels) == len(predicted_labels)
    
    # Print out the error ratio
    total_samples = len(raw_predicted_labels)
    error_ratio = (error_count / total_samples) * 100
    print(f'Errors: {error_count}/{total_samples} ({error_ratio:.2f}%)')

    return predicted_labels, errors  # Return tuple with predicted_labels and error_count

# Usage
predicted_labels, pred_errors = convert_raw_labels_to_relations(raw_predicted_labels)
print(predicted_labels[:3])

true_labels, true_errors = convert_raw_labels_to_relations(dataset_dict['test']['summary'])

config = get_config_from_stem(data_stem)
evaluator = RelationExtractorEvaluator(config=config)
df = evaluator.assess_performance_on_lists(
    true_labels_list=true_labels, pred_labels_list=predicted_labels, return_details=True,
    )
df

metric_visualizer = GranularMetricVisualizer(df=df, model_name=base_model, test_dataset_stem=data_stem)
metric_visualizer.visualize_class_metrics_distribution(df)
df_metrics_sample = metric_visualizer.visualize_class_metrics_distribution_per_class(df)
output_metrics = metric_visualizer.dump_metrics()


# # Release GPU
model.cpu()
del model
torch.cuda.empty_cache()



