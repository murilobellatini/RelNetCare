import argparse
import os
import torch
from src.processing.relation_extraction_evaluator import RelationExtractorEvaluator, GranularMetricVisualizer, load_and_process_data
from src.config import get_config_from_stem
from src.processing.bart_processing import convert_raw_labels_to_relations_bart
from src.paths import LOCAL_DATA_PATH
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

# parses input params
parser = argparse.ArgumentParser(description='Run inferences on data for BART Model evaluation')
parser.add_argument('--data_folder', type=str, default=f"{LOCAL_DATA_PATH}/processed/dialog-re-llama-11cls-rebalPairs-rwrtKeys-instrC-mxTrnCp3-skpTps-prepBART", help='Data folder path')
parser.add_argument('--model_name', type=str, default='facebook/bart-base', help='Model name')

# loads input params
args = parser.parse_args()
data_folder = args.data_folder
base_model = args.model_name
data_stem = data_folder.split('/')[-1]
model_path = f"/mnt/vdb1/murilo/models/fine-tuned/{base_model}/{data_stem}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device=",device)


# loads data for testing
dataset_dict = load_and_process_data(data_folder=data_folder, memorization_task=False, merge_train_dev=False)  

# loads model (if there is only one checkpoint - which is expected)
all_folders = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f)) if 'checkpoint' in f]
if len(all_folders) == 1:
    checkpoint_folder = all_folders[0]
    full_checkpoint_path = os.path.join(model_path, checkpoint_folder)
    print(full_checkpoint_path)
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(full_checkpoint_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(full_checkpoint_path).to(device)
else:
    raise Exception("More than one checkpoint found. Be more specific!")

# run predictions / inferences
raw_predicted_labels = []

for entry in tqdm(dataset_dict['test']):
    text = entry['text']
    inputs = tokenizer(text, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(inputs, max_length=300, do_sample=False)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw_predicted_labels.append(summary)

assert len(raw_predicted_labels) == len(dataset_dict['test'])

# extract true and predicted labels
predicted_labels, pred_errors = convert_raw_labels_to_relations_bart(raw_predicted_labels)
true_labels, true_errors = convert_raw_labels_to_relations_bart(dataset_dict['test']['summary'])

# run evaluations
config = get_config_from_stem(data_stem)
evaluator = RelationExtractorEvaluator(config=config)
df = evaluator.assess_performance_on_lists(
    true_labels_list=true_labels, pred_labels_list=predicted_labels, return_details=True,
    )
metric_visualizer = GranularMetricVisualizer(df=df, model_name=base_model, test_dataset_stem=data_stem)
metric_visualizer.visualize_class_metrics_distribution(df)
df_metrics_sample = metric_visualizer.visualize_class_metrics_distribution_per_class(df)
output_metrics = metric_visualizer.dump_metrics()

# release GPU
model.cpu()
del model
torch.cuda.empty_cache()



