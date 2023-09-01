import json
import openai
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict


from src.config import LLMTransformationConfig

config = LLMTransformationConfig()

class RelationExtractorEvaluator:
    """Evaluate the performance of a relation extraction model on a test dataset.
    
    Attributes:
        model (str): Model name to use for inference.
        openai_api (str): Base API URL for the OpenAI model.
        api_key (str): API key for accessing the model.
    """

    def __init__(self):
        self.api_key = "EMPTY"
        self.api_base = "http://localhost:8000/v1"
        self.model = "vicuna-7b-v1.1"
        # Initialize OpenAI
        openai.api_key = self.api_key
        openai.api_base = self.api_base

    @staticmethod
    def _calculate_metrics_for_entry(true_labels, predicted_labels):
        """Calculate precision, recall, and F1 score for a given set of true and predicted labels."""
        if not true_labels and not predicted_labels:  # If both are empty
            return 1, 1, 1

        if not true_labels or not predicted_labels:  # If one of them is empty
            return 0, 0, 0

        true_set = set(true_labels)
        predicted_set = set(predicted_labels)

        tp = len(true_set & predicted_set)
        fp = len(predicted_set - true_set)
        fn = len(true_set - predicted_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1

    def infer_from_model(self, dialogue, preprompt="", max_output_length=512):
        """Perform model inference given a dialogue and preprompt."""
        completion = openai.ChatCompletion.create(
            model=self.model,
            temperature=0,
            max_tokens=max_output_length,  # Set your desired max length
            messages=[{
                "role": "user",
                "content": preprompt + dialogue
            }]
        )
        return completion.choices[0].message.content

    def assess_performance_on_test_dataset(self, test_file_path, cap_size=None, return_details=False, remove_ordereddict=True):
        """Evaluate the model's performance on the provided test dataset."""
        with open(test_file_path, 'r', encoding='utf8') as fp:
            test_data = json.load(fp)

        if cap_size:
            test_data = test_data[:cap_size]

        details = []
        results_per_class = defaultdict(list)
        pbar = tqdm(test_data, dynamic_ncols=True, leave=False, position=0)

        overall_predictions = []
        overall_true = []

        result_df = pd.DataFrame(columns=["id","prompt","dialogue","true_labels","raw_inference","predicted_labels","correct_labels","wrong_labels","f1s","precision","recall","error_message"])
        errors = []
        
        total_precision, total_recall, total_f1 = 0, 0, 0
        processed_entries = 0
        errors_count = 0


        for entry in pbar:
            precision = None 
            recall = None 
            f1 = None
            dialogue = None
            true_labels = None
            raw_inference = None
            predicted_labels = None
            correct_labels = None
            wrong_labels = None
            
            prompt = "\n".join([message["value"] for message in entry["conversations"] if message["from"] == "human"])
            try:
                raw_inference = self.infer_from_model(prompt)
                predicted_relations = json.loads(raw_inference, object_pairs_hook=OrderedDict) if not raw_inference in ['', '\n'] else []
                true_relations = json.loads(entry["conversations"][1]["value"], object_pairs_hook=OrderedDict)

                predicted_labels = [str(pred_relation) for pred_relation in predicted_relations]
                true_labels = [str(true_relation) for true_relation in true_relations]

                for true_relation in true_relations:
                    results_per_class[true_relation.get('r')].append((predicted_labels, true_labels))

                precision, recall, f1 = self._calculate_metrics_for_entry(true_labels, predicted_labels)

                total_precision += precision
                total_recall += recall
                total_f1 += f1
                processed_entries += 1

                overall_predictions.extend(predicted_labels)
                overall_true.extend(true_labels)

                avg_precision = total_precision / processed_entries
                avg_recall = total_recall / processed_entries
                avg_f1 = total_f1 / processed_entries
        
                pbar.set_description(f"Avg P: {avg_precision:.1%} | Avg R: {avg_recall:.1%} | Avg F1: {avg_f1:.1%} | Errors: {errors_count}/{len(test_data)} ({errors_count/len(test_data):.0%})")

                # Calculate correct and wrong labels
                correct_labels = list(set(true_labels) & set(predicted_labels))
                wrong_labels = list(set(predicted_labels) - set(true_labels))
                
                dialogue = prompt.split('Input:')[-1].replace('Output:','')

                if return_details:
                    detail = {
                        "id": entry['id'],
                        "prompt": prompt.replace(config.preprompt, ''),
                        "dialogue": dialogue,
                        "true_labels": true_labels,
                        "raw_inference": raw_inference,
                        "predicted_labels": predicted_labels,
                        "correct_labels": correct_labels,
                        "wrong_labels": wrong_labels,
                        "f1s": f1,
                        "precision": precision,
                        "recall": recall,
                        "error_message": ''

                    }
                    details.append(detail)
                    result_df.loc[len(result_df.index)] = detail
    

            except Exception as e:
                errors_count += 1

                errors.append(f"{entry['id']}: {e}")
                                            
                def get_value_from_locals(var_name, local_vars, transform_func=str, default_value=None):
                    value = local_vars.get(var_name, default_value)
                    if isinstance(value, (list, tuple)):
                        return [transform_func(item) for item in value]
                    return value

                local_vars = locals()  # Capture local variables where the function will be called

                # Inside your loop
                error_detail = {
                    "id": entry['id'],
                    "prompt": get_value_from_locals('prompt', local_vars),
                    "dialogue": get_value_from_locals('dialogue', local_vars),
                    "true_labels": get_value_from_locals('true_relations', local_vars),
                    "raw_inference": get_value_from_locals('raw_inference', local_vars),
                    "predicted_labels": get_value_from_locals('predicted_relations', local_vars),
                    "correct_labels": get_value_from_locals('correct_labels', local_vars),
                    "wrong_labels": get_value_from_locals('wrong_labels', local_vars),
                    "f1s": get_value_from_locals('f1', local_vars, default_value=0),
                    "precision": get_value_from_locals('precision', local_vars, default_value=0),
                    "recall": get_value_from_locals('recall', local_vars, default_value=0),
                    "error_message": str(e)
                }

                result_df.loc[len(result_df.index)] = error_detail
                


        output_path = test_file_path.replace('.json', '')
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if remove_ordereddict:
            for col in result_df.columns:
                if 'labels' in col:
                    result_df[col] = result_df[col].apply(lambda x: [xi.replace('OrderedDict', '') for xi in x] if x is not None else None)

                
        output_path = f"{output_path}_{current_time}.xlsx"
        result_df.to_excel(output_path, index=False)
        print(f"\nScript successfully executed!")
        print(f"Avg P: {avg_precision:.1%} | Avg R: {avg_recall:.1%} | Avg F1: {avg_f1:.1%} | Errors: {errors_count}/{len(test_data)} ({errors_count/len(test_data):.0%})")
        print(f"# INFERENCE REPORT\n{output_path}\n")

            
        overall_precision, overall_recall, overall_f1 = self._calculate_metrics_for_entry(overall_true, overall_predictions)

        per_class_results = {}
        for relation, labels_list in results_per_class.items():
            preds, trues = [], []
            for preds_labels, true_labels in labels_list:
                preds.extend(preds_labels)
                trues.extend(true_labels)

            precision, recall, f1 = self._calculate_metrics_for_entry(trues, preds)

            per_class_results[relation] = {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

        result = {
            "overall": {
                "precision": overall_precision,
                "recall": overall_recall,
                "f1": overall_f1
            },
            "per_class": per_class_results
        }

        if return_details:
            result["details"] = details

        return result_df

class FileManager:
    """Handle reading and writing of files."""
    
    @staticmethod
    def read_json_file(file_path):
        """Read a JSON file and return its content."""
        with open(file_path, 'r', encoding='utf8') as fp:
            return json.load(fp)
    
    # Any other file-related functions can be added here

class RelationGranularMetrics(RelationExtractorEvaluator):
    def __init__(self, df, ontology):
        self.df = df
        self.ontology = ontology
        self.result = {}
    
    def aggregate_by_relation(self, group):
        metrics_by_relation = {}
        all_true_labels = group['true_labels'].iloc[0] if any(group['true_labels']) else ["Null-Relation"]
        all_predicted_labels = group['predicted_labels'].iloc[0] if any(group['predicted_labels']) else ["Null-Relation"]
        
        for r in self.ontology:
            true_labels = [str(x) for x in all_true_labels]
            predicted_labels = [str(x) for x in all_predicted_labels if r in x]
            
            if not true_labels and not predicted_labels:
                metrics_by_relation[r] = {'precision': None, 'recall': None, 'f1': None}
            else:
                precision, recall, f1 = self._calculate_metrics_for_entry(true_labels, predicted_labels)
                metrics_by_relation[r] = {'precision': precision, 'recall': recall, 'f1': f1}
        
        return metrics_by_relation
    
    def process(self):
        grouped = self.df.groupby('id')
        for name, group in grouped:
            self.result[name] = self.aggregate_by_relation(group)
        return self.result
    
    def to_dataframe(self):
        chart_df = pd.DataFrame.from_dict({(i, j): self.result[i][j] 
                                for i in self.result.keys() 
                                for j in self.result[i].keys()}, 
                                orient='index')
        return chart_df
    
    def plot_metrics(self, chart_df):
        agg_stats = chart_df.groupby(level=1).agg(['mean', 'std'])
        fig, ax = plt.subplots()
        agg_stats = agg_stats.sort_values(by=('f1', 'mean'), ascending=True)
        agg_stats.xs('mean', axis=1, level=1).plot(kind='barh', xerr=agg_stats.xs('std', axis=1, level=1), ax=ax, figsize=(3,5))
        plt.xlabel('Metrics')
        plt.title('Average and Stddev for Relations')
        plt.xlim(0, 1)
        plt.legend(loc='lower right')
        plt.show()
