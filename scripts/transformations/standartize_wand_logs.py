import shutil
import json
import os
import wandb
from src.paths import LOCAL_DATA_PATH

def get_run_data(api, project, run_id):
    run = api.run(f"{project}/{run_id}")
    logs_df = run.history()
    final_data = logs_df.iloc[-1].to_dict()
    return final_data, run.config

def prepare_new_data(final_data):
    new_data = {
        "micro_avg": {"precision": None, "recall": None, "f1": None},
        "macro_avg": {"precision": None, "recall": None, "f1": None},
        "per_class": {}
    }

    total_precision, total_recall, total_f1 = 0.0, 0.0, 0.0

    for key, value in final_data.items():
        if 'class_' in key:
            key = key.replace('class_no_relation', 'class_per:no_relation')
            key = key.replace('class_with_relation', 'class_per:with_relation')
            parts = key.split('/')
            try:
                class_name = parts[0].split(':')[1]
            except IndexError:
                print(f"key={key}")
            metric = parts[-1]
            if class_name not in new_data["per_class"]:
                new_data["per_class"][class_name] = {}
            new_data["per_class"][class_name][metric] = value

            total_precision += value if metric == 'precision' else 0
            total_recall += value if metric == 'recall' else 0
            total_f1 += value if metric == 'f1' else 0

    total_classes = len(new_data['per_class'])
    new_data['micro_avg']['precision'] = total_precision / total_classes
    new_data['micro_avg']['recall'] = total_recall / total_classes
    new_data['micro_avg']['f1'] = total_f1 / total_classes

    return new_data

def write_json(data, output_dir, file_name='class_metrics.json'):
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir / file_name, encoding='utf-8', mode='w') as fp:
        json.dump(data, fp)

def main(project, run_id):
    api = wandb.Api()

    final_data, run_config = get_run_data(api, project, run_id)
    new_data = prepare_new_data(final_data)
    new_data["macro_avg"]["f1"] = final_data["f1"]
    new_data['exp_group'] = run_config['exp_group']

    model_name = run_config['vocab_file'].split('downloaded/')[-1].replace('/vocab.txt','')
    input_data = run_config['data_dir'].split('data/')[-1].replace('raw/', 'raw-').replace('processed/', '')

    output_dir = LOCAL_DATA_PATH / 'reports' / input_data / model_name
    print("output_dir=",output_dir)

    data_dir = run_config['data_dir']
    readme_path = f'{data_dir}/README.md'

    write_json(new_data, output_dir)

    if os.path.exists(readme_path):
        shutil.copy(readme_path, output_dir)
    else:
        print(f"README.md for not found on {readme_path}")


if __name__ == "__main__":
    project, run_id = "mbellatini/RelNetCare", "t4bgqp2s"
    main(project, run_id)
