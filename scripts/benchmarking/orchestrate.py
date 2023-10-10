import os
import yaml
import subprocess
from dotenv import load_dotenv

# Load the YAML config file
config_path = os.getenv("BENCHMARKING_CONFIG_PATH")
with open(config_path, 'r') as stream:
    config = yaml.safe_load(stream)

# Pretty print the config
print("Experiment Group:")
print(f"  - {config['exp_group'][0]}")
print("\nModels:")
for model in config['models']:
    print(f"  - {model}")
print("\nDatasets:")
for dataset in config['datasets']:
    print(f"  - {dataset}")

# Prompt user for pipeline choice
choice = input("\nChoose the desired pipeline:\n1. Train and Infer/Test\n2. Only Train\n3. Only Infer/Test\n> ")

# Loop through models and datasets to run the scripts
for model in config['models']:
    for dataset in config['datasets']:
        # Reset data_folder for each dataset
        data_folder = f"/home/murilo/RelNetCare/data/processed/{dataset}"
        
        # Append 'prepBART' if model name contains 'BART' or 'Rebel'
        if 'BART' in model or 'Rebel' in model:
            data_folder += '-prepBART'
            
        script_path_train = f"/home/murilo/RelNetCare/scripts/benchmarking/model_scripts/{model}/train.py"
        script_path_infer = f"/home/murilo/RelNetCare/scripts/benchmarking/model_scripts/{model}/infer_eval.py"
        
        if choice in ["1", "2"]:
            # Skip training for Rebel
            if model not in ("RebelBaseline", "ensemble"):
                print("Action: Training.")
                print(f"• Model: {model}")
                print(f"• Dataset: {dataset}")
                print(f"• Experiment Group: {config['exp_group'][0]}")
                subprocess.run(["python", script_path_train, "--data_folder", data_folder, "--exp_group", config['exp_group'][0]])
        
        if choice in ["1", "3"]:
            print("Action: Running Inference Evaluation.")
            print(f"• Model: {model}")
            print(f"• Dataset: {dataset}")
            subprocess.run(["python", script_path_infer, "--data_folder", data_folder])

