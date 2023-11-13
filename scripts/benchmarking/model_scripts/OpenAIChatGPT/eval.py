import os
import json
import pathlib as pl
from src.config import get_config_from_stem
from src.processing.relation_extraction_evaluator import RelationExtractorEvaluator, GranularMetricVisualizer, load_and_process_data, parse_json_objects
import pandas as pd
from ast import literal_eval


model_name = 'gpt-3.5-turbo'
exp_group = 'test'
# data_path = pl.Path('/home/murilo/RelNetCare/data/processed/chat-gpt-predictions/dialog-re-37cls-with-no-relation-undersampled-llama-clsTskOnl')
data_path = pl.Path('/home/murilo/RelNetCare/data/processed/chat-gpt-predictions/dialog-re-12cls-with-no-relation-undersampled-llama')
data_stem = data_path.name
cls_task_only = 'clsTskOnl' in data_stem

data = []
for f in os.listdir(data_path):
    if f.endswith('.json'):
        with open(data_path / f, encoding='utf8') as fp:
            tmp = json.load(fp)
        data.append(tmp)
        
df = pd.DataFrame(data)

true_labels = df.ground_truth_response.apply(eval)
# extract true and predicted labels
if cls_task_only:
    predicted_labels = [[x] for x in df.open_ai_predicted_response.tolist()]
    true_labels = [literal_eval(x) for x in df.ground_truth_response.tolist()]
else:
    predicted_labels = [parse_json_objects(x) for x in df.open_ai_predicted_response.tolist()]
    true_labels = [parse_json_objects(x) for x in df.ground_truth_response.tolist()]

config = get_config_from_stem(data_stem)
evaluator = RelationExtractorEvaluator(config=config)
# run evaluations
df = evaluator.assess_performance_on_lists(
    true_labels_list=true_labels, pred_labels_list=predicted_labels, return_details=True,
    )
metric_visualizer = GranularMetricVisualizer(df=df, model_name=model_name, test_dataset_stem=data_stem, cls_task_only=cls_task_only, exp_group=exp_group)
metric_visualizer.visualize_class_metrics_distribution(df)
df_metrics_sample = metric_visualizer.visualize_class_metrics_distribution_per_class(df)
output_metrics = metric_visualizer.dump_metrics()

