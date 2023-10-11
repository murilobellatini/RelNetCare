from src.infering import CustomTripletExtractor
from src.processing.relation_extraction_evaluator import RelationExtractorEvaluator, GranularMetricVisualizer
from src.paths import LOCAL_DATA_PATH
from src.config import get_config_from_stem
from dotenv import load_dotenv
from ast import literal_eval
from tqdm import tqdm
import argparse
import torch
import json

load_dotenv()

# parses input params
parser = argparse.ArgumentParser(description='Run inferences on data for Rebel Model evaluation')
parser.add_argument('--data_folder', type=str, default=f"{LOCAL_DATA_PATH}/processed/dialog-re-llama-11cls-rebalPairs-rwrtKeys-instrC-mxTrnCp3-skpTps", help='Data folder path')
parser.add_argument('--model_name', type=str, default='custom/ensemble', help='Model name')

# loads input params
args = parser.parse_args()
data_folder = args.data_folder
data_stem = data_folder.split('/')[-1]
base_model = args.model_name
test_file_path = f"{LOCAL_DATA_PATH}/processed/{data_stem}/{data_stem}-test.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device=",device)

# loads data for testing
print('Loading test data...')
with open(test_file_path, encoding='utf8') as fp:
    test_data = json.load(fp)

dialogues = [d['conversations'][0]['value'].split('Dialogue: \n')[-1] for d in test_data]
dialogues = [literal_eval(d) for d in dialogues ]

# loads model
print('Loading model...')
extractor = CustomTripletExtractor(apply_coref_resolution=False)

# run inference 
predicted_labels = []

print('Running inferences...')
for d in tqdm(dialogues, desc="Predicting"):
    raw_triplets = extractor.extract_triplets(d)
    out_triplets = [{'subject': t['x'], 'relation': t['r'][0].split(':')[-1], 'object': t['y']} for t in raw_triplets if 'r' in t.keys()]
    predicted_labels.append(out_triplets )

# extract true labels
true_labels = [literal_eval(d['conversations'][1]['value']) for d in test_data]

assert len(true_labels) == len(predicted_labels)

# run evaluations
print('Running evaluation...')
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
print('Done!')
