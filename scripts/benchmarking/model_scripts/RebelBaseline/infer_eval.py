import argparse
import torch
from src.processing.relation_extraction_evaluator import RelationExtractorEvaluator, GranularMetricVisualizer, load_and_process_data
from src.processing.bart_processing import convert_raw_labels_to_relations_bart
from src.processing.rebel_processing import run_inference_and_extract_triplets, get_model
from src.paths import LOCAL_DATA_PATH
from src.config import get_config_from_stem
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

# parses input params
parser = argparse.ArgumentParser(description='Run inferences on data for Rebel Model evaluation')
parser.add_argument('--data_folder', type=str, default=f"{LOCAL_DATA_PATH}/processed/dialog-re-llama-11cls-rebalPairs-rwrtKeys-instrC-mxTrnCp3-skpTps-prepBART", help='Data folder path')
parser.add_argument('--model_name', type=str, default='Babelscape/rebel-large', help='Model name')
parser.add_argument('--batch_size', type=int, default=128, help='Inference batch size')
parser.add_argument('--exp_group', type=str, default='TestPipeline', help='Experiment group name')

# loads input params
args = parser.parse_args()
data_folder = args.data_folder
base_model = args.model_name
batch_size = args.batch_size
data_stem = data_folder.split('/')[-1]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device=",device)

# loads data for testing
print('Loading test data...')
dataset_dict = load_and_process_data(data_folder=data_folder, memorization_task=False, merge_train_dev=False)  

# loads model
print('Loading model...')
model = get_model(device=device, base_model=base_model)

# run inference 
predicted_labels = []
num_batches = len(dataset_dict['test']['text']) // batch_size

print('Running inferences...')
for i in tqdm(range(num_batches), desc="Predicting"):
    batched_input = dataset_dict['test']['text'][i * batch_size: (i + 1) * batch_size]
    predicted_labels.extend(run_inference_and_extract_triplets(model, batched_input))

print('Handling samples missing samples from batch')
if len(dataset_dict['test']['text']) % batch_size:
    leftover_start = num_batches * batch_size
    leftover_input = dataset_dict['test']['text'][leftover_start:]
    predicted_labels.extend(run_inference_and_extract_triplets(model, leftover_input))
    
# extract true labels
true_labels, true_errors = convert_raw_labels_to_relations_bart(dataset_dict['test']['summary'])

assert len(true_labels) == len(predicted_labels)

# run evaluations
print('Running evaluation...')
config = get_config_from_stem(data_stem)
evaluator = RelationExtractorEvaluator(config=config)
df = evaluator.assess_performance_on_lists(
    true_labels_list=true_labels, pred_labels_list=predicted_labels, return_details=True,
    )
metric_visualizer = GranularMetricVisualizer(df=df, model_name=base_model, test_dataset_stem=data_stem, exp_group=args.exp_group)
metric_visualizer.visualize_class_metrics_distribution(df)
df_metrics_sample = metric_visualizer.visualize_class_metrics_distribution_per_class(df)
output_metrics = metric_visualizer.dump_metrics()

# release GPU
print('Removing model from memory...')
del model
torch.cuda.empty_cache()

print('Done!')
