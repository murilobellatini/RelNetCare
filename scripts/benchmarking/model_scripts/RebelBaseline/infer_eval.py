import argparse
import torch
from src.processing.relation_extraction_evaluator import RelationExtractorEvaluator, GranularMetricVisualizer, load_and_process_data
from src.processing.bart_processing import convert_raw_labels_to_relations_bart
from src.processing.rebel_processing import run_inference_and_extract_triplets, get_model
from src.config import get_config_from_stem
from tqdm import tqdm

# parses input params
parser = argparse.ArgumentParser(description='Run inferences on data for Rebel Model evaluation')
parser.add_argument('--data_folder', type=str, default="/home/murilo/RelNetCare/data/processed/dialog-re-llama-11cls-rebalPairs-rwrtKeys-instrC-mxTrnCp3-skpTps-prepBART", help='Data folder path')
parser.add_argument('--model_name', type=str, default='Babelscape/rebel-large', help='Model name')

# loads input params
args = parser.parse_args()
data_folder = args.data_folder
base_model = args.model_name
data_stem = data_folder.split('/')[-1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device=",device)

# loads data for testing
dataset_dict = load_and_process_data(data_folder=data_folder, memorization_task=False, merge_train_dev=False)  

# loads model
model = get_model(device, base_model=base_model)

# run inference 
predicted_labels = []
for input_ in tqdm(dataset_dict['test']['text'], desc="Predicting"):
    predicted_labels.append(run_inference_and_extract_triplets(input_, device, model))

# extract true labels
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