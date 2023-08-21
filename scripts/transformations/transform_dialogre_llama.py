"""
Data Transformation Script for DialogRE

This script transforms the DialogRE dataset into a format suitable for instructing LLMs (Language Learning Models) 
to perform relation extraction tasks, facilitating fine-tuning on the LoRA LLaMa scripts.

Usage:
    python script_name.py --max_turns 10 --max_speakers 2

Parameters:
    --max_turns:      Maximum number of dialogue turns to be processed (default: None).
    --max_speakers:   Maximum number of distinct speakers in a dialogue (default: None).

Output:
    A `.jsonl` file with structured data ready for LLM fine-tuning.
"""

import os
import json
import argparse
from src.config import LLMTransformationConfig
    
class DataTransformer:
    @staticmethod
    def transform_data_to_llm_format(data, config, start_idx=0):
        """Transform the given data into LLM ready dialogue format."""
        preprompt = config.preprompt
        skip_relations = config.skip_relations
        max_turns = config.max_turns
        max_speakers = config.max_speakers
        
        new_data = []
        identity_counter = start_idx

        for conversation, triples in data:
            # Filtering dialogues based on max_turns criteria
            if max_turns is not None and len(conversation) > max_turns:
                continue

            # Extracting speaker names from each dialogue turn and checking based on max_speakers criteria
            if max_speakers is not None:
                speakers = set([turn.split(":")[0].strip() for turn in conversation])
                if len(speakers) > max_speakers:
                    continue


            triples_text = [
                {
                    "x": triple["x"],
                    "x_type": triple["x_type"],
                    "r": triple["r"][0].split(':')[-1],
                    "y": triple["y"],
                    "y_type": triple["y_type"]
                }
                for triple in triples
                if triple["r"] and triple["r"][0].split(':')[-1] not in skip_relations
            ]

            conversation_entry = {
                "id": f"identity_{identity_counter}",
                "conversations": [
                    {"from": "human", #tried: human
                    "value": preprompt + "\n".join(conversation)},
                    {"from": "gpt", #tried: gpt
                    "value": str(json.dumps(triples_text))}
                ]
            }

            identity_counter += 1

            if triples_text:
                new_data.append(conversation_entry)

        return new_data

    @staticmethod
    def process_and_save_data(config):
        """Process the input data and save in the specified format."""
        file_sets=config.file_sets
        input_dir=config.input_dir
        output_dir=config.output_dir
        skip_relations=config.skip_relations
        preprompt=config.preprompt
        total_relation_count=config.total_relation_count
        max_turns=config.max_turns
        max_speakers=config.max_speakers

        output_data = []
        for files in file_sets:
            data_folder_name = output_dir.split('/')[-1]
            dataset_name = f"{data_folder_name}-{'-'.join(files)}"
            last_data_idx = 0
            new_format = []

            DataManager.create_directory_if_not_exists(output_dir)

            for f in files:
                input_data_path = os.path.join(input_dir, f'{f}.json')
                
                # Check if file exists
                if os.path.exists(input_data_path):
                    with open(input_data_path, encoding='utf8') as fp:
                        data = json.load(fp)

                    new_format.extend(DataTransformer().transform_data_to_llm_format(data, config, last_data_idx))
                    last_data_idx = len(new_format)

            output_data_path = os.path.join(output_dir, f'{dataset_name}.json')
            with open(output_data_path, 'w', encoding='utf8') as fp:
                json.dump(new_format, fp)

            print(files, len(new_format))
            
            output_data.append(new_format)
            
        return output_data

class DataManager:
    @staticmethod
    def create_directory_if_not_exists(directory_path):
        """Create a directory if it does not already exist."""
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            
    @staticmethod
    def read_json_file(file_path):
        with open(file_path, encoding='utf8') as fp:
            return json.load(fp)

    @staticmethod
    def write_json_file(data, file_path):
        with open(file_path, 'w', encoding='utf8') as fp:
            json.dump(data, fp)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform data to a desired format.")
    parser.add_argument("--max_turns", type=int, default=None, help="Maximum number of turns.")
    parser.add_argument("--max_speakers", type=int, default=2, help="Maximum number of speakers.")
    args = parser.parse_args()

    # Create a Config instance with the parsed arguments
    config = LLMTransformationConfig(args.max_turns, args.max_speakers)

    # Process and save data
    DataTransformer.process_and_save_data(config)
