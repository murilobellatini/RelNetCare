from src.paths import LOCAL_DATA_PATH
from src.config import get_config_from_stem
from src.processing.relation_extraction_evaluator import RelationExtractorEvaluator, GranularMetricVisualizer, load_and_process_data

import os
import time
import argparse
import subprocess
from tqdm import tqdm
from ast import literal_eval
from dotenv import load_dotenv


load_dotenv()

# Create an argument parser
parser = argparse.ArgumentParser(description='Train LLaMA Lora model')

# Add arguments
parser.add_argument('--data_folder', type=str, default=f"{LOCAL_DATA_PATH}/processed/dialog-re-llama-11cls-rebalPairs-rwrtKeys-instrC-mxTrnCp3-skpTps", help='Data folder path')
parser.add_argument('--model_name', type=str, default='llama-7B-hf', help='Model name')
parser.add_argument('--merge_dev_train', type=bool, default=True, help='Model name')

# Parse the arguments
args = parser.parse_args()
data_stem = args.data_folder.split('/')[-1]

# Initialization
model_output_dir= f"{os.environ['MODEL_DIR']}/fine-tuned/{args.model_name}/{data_stem}"

# Launch processes
controller_proc = subprocess.Popen(["python", "-m", "fastchat.serve.controller"])
model_worker_proc = subprocess.Popen(["python", "-m", "fastchat.serve.model_worker", "--model-name", "vicuna-7b-v1.1", "--model-path", model_output_dir])

# Give services some time to start
time.sleep(15)

# Start the OpenAI server
openai_server_proc = subprocess.Popen(["python", "-m", "fastchat.serve.openai_api_server", "--host", "localhost", "--port", "8000"])

# Give services some time to start
time.sleep(30)

# run predictions / inferences
raw_predicted_labels = []
config = get_config_from_stem(data_stem)
evaluator = RelationExtractorEvaluator(config=config)
test_file_path = f"{LOCAL_DATA_PATH}/processed/{data_stem}/{data_stem}-test.json"
df = evaluator.assess_performance_on_test_dataset(test_file_path, return_details=True)


# extract true and predicted labels
predicted_labels = [literal_eval(x) for x in df.predicted_labels.tolist()]
true_labels = [literal_eval(x) for x in df.true_labels.tolist()]

# run evaluations
df = evaluator.assess_performance_on_lists(
    true_labels_list=true_labels, pred_labels_list=predicted_labels, return_details=True,
    )
metric_visualizer = GranularMetricVisualizer(df=df, model_name=args.model_name, test_dataset_stem=data_stem)
metric_visualizer.visualize_class_metrics_distribution(df)
df_metrics_sample = metric_visualizer.visualize_class_metrics_distribution_per_class(df)
output_metrics = metric_visualizer.dump_metrics()


# Terminate background processes
controller_proc.terminate()
model_worker_proc.terminate()
openai_server_proc.terminate()
