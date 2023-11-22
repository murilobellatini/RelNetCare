
import re
import torch
import language_tool_python

import os
import re
import signal
import threading
import psutil
import wandb
import functools
import pandas as pd
import json
from sklearn.metrics import f1_score, precision_score, recall_score


class WandbKiller:
    def __init__(self):
        self.timer = threading.Timer(60, self.force_finish_wandb)

    def force_finish_wandb(self):
        try:
            with open(os.path.join(os.path.dirname(__file__), '../wandb/latest-run/logs/debug-internal.log'), 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Failed to open log file: {e}")
            return

        if not lines:
            print("Log file is empty.")
            return

        last_line = lines[-1]
        match = re.search(r'(HandlerThread:|SenderThread:)\s*(\d+)', last_line)
        if match:
            pid = int(match.group(2))
            try:
                p = psutil.Process(pid)
                if 'wandb' in ' '.join(p.cmdline()):
                    print(f'wandb pid: {pid}')
                else:
                    print('Process is not a wandb process.')
                    return
            except psutil.NoSuchProcess:
                print('Process does not exist.')
                return
        else:
            print('Cannot find wandb process-id.')
            return

        try:
            os.kill(pid, signal.SIGKILL)
            print(f"Process with PID {pid} killed successfully.")
        except OSError:
            print(f"Failed to kill process with PID {pid}.")

    def try_finish_wandb(self):
        self.timer.start()
        wandb.finish()
        self.timer.cancel()


def to_camel_case(string):
    words = string.split('-')
    camel_case = ''.join([w.capitalize() for w in words])
    return camel_case


def handle_exceptions(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            return None
    return wrapper


def get_value_from_locals(var_name, local_vars, transform_func=str, default_value=None):
    value = local_vars.get(var_name, default_value)
    if isinstance(value, (list, tuple)):
        return [transform_func(item) for item in value]
    return value



def fix_cls_metrics_dump(data_stem, model_name, reports_path="/home/murilo/RelNetCare/data/reports"):
    data_path = f"{reports_path}/{data_stem}/{model_name}"
    # Read the JSON lines file
    df = pd.read_json(f"{data_path}/report.json", lines=True)

    # Extract true_labels and predicted_labels
    y_true = df['true_labels'].tolist()
    y_pred = df['predicted_labels'].tolist()

    # Flatten the lists
    y_true_flat = [label for sublist in y_true for label in sublist]
    y_pred_flat = [label for sublist in y_pred for label in sublist]

    # Compute Micro and Macro metrics
    metrics = {
        "micro_avg": {
            "precision": precision_score(y_true_flat, y_pred_flat, average='micro'),
            "recall": recall_score(y_true_flat, y_pred_flat, average='micro'),
            "f1": f1_score(y_true_flat, y_pred_flat, average='micro')
        },
        "macro_avg": {
            "precision": precision_score(y_true_flat, y_pred_flat, average='macro'),
            "recall": recall_score(y_true_flat, y_pred_flat, average='macro'),
            "f1": f1_score(y_true_flat, y_pred_flat, average='macro')
        },
        "per_class": {},
        "exp_group": ""
    }

    # Compute per-class metrics
    unique_labels = set(y_true_flat + y_pred_flat)
    for label in unique_labels:
        y_true_binary = [1 if l == label else 0 for l in y_true_flat]
        y_pred_binary = [1 if l == label else 0 for l in y_pred_flat]
        metrics["per_class"][label.replace("['","").replace("']","")] = {
            "precision": precision_score(y_true_binary, y_pred_binary),
            "recall": recall_score(y_true_binary, y_pred_binary),
            "f1": f1_score(y_true_binary, y_pred_binary)
        }

    # Dump metrics to JSON
    with open(f"{data_path}/class_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Metrics saved to metrics.json")
    
    
def extract_bot_reply(outputs, bot_name='Adele'):
    # Extracts raw response from full generated text
    
    raw_text = outputs[0]["generated_text"]

    patterns = ('<|im_start|>assistant\n', '<|assistant|>\n')
    
    for p in patterns:
        if p in raw_text:
            break
    
    raw_response = outputs[0]["generated_text"].split(p)[-1]

    # Define the pattern to extract the bot's reply
    pattern = rf'{bot_name}: ([^\(\n]*)'

    # Find all matches of the pattern in the text
    matches = re.findall(pattern, raw_response)

    # Return the first match stripped of leading/trailing whitespace, or None if no match
    return matches[0].strip() if matches else raw_response


def correct_text(input_text, language='de-DE'):
    # Initialize the LanguageTool object with the specified language
    tool = language_tool_python.LanguageTool(language)

    # Check the text
    matches = tool.check(input_text)

    # Generate corrected text
    corrected_text = language_tool_python.utils.correct(input_text, matches)

    return corrected_text