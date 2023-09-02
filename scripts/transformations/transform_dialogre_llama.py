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
        replace_skipped_with_others = config.replace_skipped_with_others
        
        
        new_data = []
        identity_counter = start_idx

        # If the parse_subdialogues flag is enabled, the data is extended by breaking down each dialogue into subdialogues. 
        # This allows for more granular context within the dialogue_relation pairs, capturing relationships based on the 
        # subdialogue content, and excluding relationships where the entities are not mentioned within the subdialogue.
        # @todo: Validate data manually or with GPT-4.
        if config.parse_subdialogues is True:
            data = DataTransformer.parse_subdialogues(data)

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
                    "r": "others" if replace_skipped_with_others and triple["r"][0].split(':')[-1] in skip_relations else triple["r"][0].split(':')[-1],
                    "y": triple["y"],
                    "y_type": triple["y_type"]
                }
                for triple in triples
                if replace_skipped_with_others or triple["r"][0].split(':')[-1] not in skip_relations
            ]
            
            
            if config.rewrite_keys:
                updated_triples_text = []
                for triple in triples_text:
                    updated_triple = {config.replacement_dict.get(k, k): v for k, v in triple.items()}
                    updated_triples_text.append(updated_triple)
                triples_text = updated_triples_text

            conversation_entry = {
                "id": f"identity_{identity_counter}",
                "conversations": [
                    {"from": "human", #tried: human
                    "value": preprompt + f"\n{json.dumps(conversation, indent=1, ensure_ascii=False)}" + "\n\nOutput:"},
                    {"from": "gpt", #tried: gpt
                    "value": str(json.dumps(triples_text, ensure_ascii=False))}
                ]
            }
            
            if config.cls_task_only:
                conversation_entry = []
                for i, t in enumerate(triples_text):
                    entry = {
                    "id": f"identity_{identity_counter}_{i:03d}",
                    "conversations": [
                        {"from": "human", #tried: human
                        "value": preprompt + f"{json.dumps(conversation, indent=1, ensure_ascii=False)}" + f"\n\nSubject: {t['x']} ({t['x_type']})\n" + f"Object: {t['y']} ({t['y_type']})" + "\nRelation:"},
                        {"from": "gpt", #tried: gpt
                        "value": str(t['r'])}
                    ]
            }
                    conversation_entry.append(entry)

            identity_counter += 1

            if not config.skip_empty_triples or triples_text:
                if config.cls_task_only:
                    new_data.extend(conversation_entry)
                else:
                    new_data.append(conversation_entry)
                
                
        if not config.cls_task_only:
            # Separate dialogues with empty and non-empty triples
            relation_key = "relation" if config.rewrite_keys else "r"
            dialogues_with_triples = [entry for entry in new_data if entry['conversations'][1]['value'] != '[]' and not all(triple[relation_key] == "others" for triple in json.loads(entry['conversations'][1]['value']))]
            dialogues_without_triples = [entry for entry in new_data if entry['conversations'][1]['value'] == '[]' or all(triple[relation_key] == "others" for triple in json.loads(entry['conversations'][1]['value']))]

            # Balance the dialogues by keeping only as many empty dialogues as there are non-empty dialogues
            if config.balance_empty_dialogues is True:
                dialogues_without_triples = dialogues_without_triples[:len(dialogues_with_triples)]

            # Cap rebalancing according to the average count of dialgues with triples per relation
            if config.rebalance_empty_dialogues is True:
                dialogues_without_triples = dialogues_without_triples[:int(3*len(dialogues_with_triples)/len(config.allowed_relations))]

            # Combine and reorder according to the id key
            new_data = dialogues_with_triples + dialogues_without_triples
            new_data.sort(key=lambda x: int(x['id'].split('_')[1]))

        return new_data

    @staticmethod
    def parse_subdialogues(data):
        extended_data = []
        for conversation, triples in data:
            for i in range(1, len(conversation) + 1):
                sub_conversation = conversation[:i]
                sub_triples = [triple for triple in triples if all(entity in ' '.join(sub_conversation) for entity in [triple['x'], triple['y']])]
                extended_data.append((sub_conversation, sub_triples))
        return extended_data


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
                    with open(input_data_path, encoding='utf-8') as fp:
                        data = json.load(fp)

                    new_format.extend(DataTransformer().transform_data_to_llm_format(data, config, last_data_idx))
                    last_data_idx = len(new_format)

            output_data_path = os.path.join(output_dir, f'{dataset_name}.json')
            with open(output_data_path, 'w', encoding='utf-8') as fp:
                json.dump(new_format, fp, ensure_ascii=False)

            print(files, len(new_format))
            
            output_data.append(new_format)
            
        DataTransformer.generate_readme(output_dir, file_sets, output_data, config)

        return output_data

    @staticmethod
    def generate_readme(output_dir, file_sets, output_data, config):
        readme_path = os.path.join(output_dir, 'README.md')
        
        relation_counts = {}
        relation_key = "relation" if config.rewrite_keys else "r"
        # Collecting all relation classes
        for data_sets in output_data:
            for sample in data_sets:
                label = sample['conversations'][1]['value']
                relations = [label] if config.cls_task_only else json.loads(label)
                if not relations:
                    relation_counts['[NULL_RELATIONS]'] = relation_counts.get('[NULL_RELATIONS]', 0) + 1
                else:
                    for r in relations:
                        relation = r if config.cls_task_only else r[relation_key]
                        relation_counts[relation] = relation_counts.get(relation, 0) + 1

        total_relations = sum(relation_counts.values())


        with open(readme_path, 'w') as fp:
            fp.write("# Data Transformation Details\n\n")
            fp.write(f"- **Dataset Name**: {config.output_dir.split('/')[-1]}\n")
            fp.write("## Parameters Used:\n")
            fp.write(f"- **Allowed Relation Count**: {len(config.allowed_relations)}\n")
            fp.write(f"- **Max Speakers**: {config.max_speakers}\n")
            fp.write(f"- **Max Turns**: {config.max_turns}\n")
            fp.write(f"- **Balance Null-Relation Dialogues**: {config.balance_empty_dialogues}\n")
            fp.write(f"- **Rebalance Null-Relation Dialogues**: {config.balance_empty_dialogues}\n")
            fp.write(f"- **Replace Skipped With Others**: {config.replace_skipped_with_others}\n")
            fp.write(f"- **Allowed Relations**: `{sorted(config.allowed_relations)}`\n")
            fp.write(f"- **Prompt Instruction (_Variant {config.instruction_type}_)**: \n```\n{config.preprompt}\n```\n")
            fp.write("\n## Files and Dialogue Counts:\n")
            for file_set, new_format in zip(file_sets, output_data):
                fp.write(f"- **File Name**: {file_set}, **Count**: {len(new_format)}\n")
                
            fp.write("\n## Relation Counts and Percentages:\n")
            fp.write("| Relation | Count | Percentage |\n")
            fp.write("|----------|-------|------------|\n")
            for relation, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_relations) * 100
                fp.write(f"| {relation} | {count} | {percentage:.1f}% |\n")
                
                
class DataManager:
    @staticmethod
    def create_directory_if_not_exists(directory_path):
        """Create a directory if it does not already exist."""
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            
    @staticmethod
    def read_json_file(file_path):
        with open(file_path, encoding='utf-8') as fp:
            return json.load(fp)

    @staticmethod
    def write_json_file(data, file_path):
        with open(file_path, 'w', encoding='utf-8') as fp:
            json.dump(data, fp, ensure_ascii=False)

    
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Transform data to a desired format.")
    # parser.add_argument("--max_turns", type=int, default=None, help="Maximum number of turns.")
    # parser.add_argument("--max_speakers", type=int, default=None, help="Maximum number of speakers.")
    # parser.add_argument("--balance_empty_dialogues", type=bool, default=True, help="Balance empty/non-empty dialogue-relations pairs.")
    # parser.add_argument("--replace_skipped_with_others", type=bool, default=False, help="Replace non-focus relations with 'others'.")
    # args = parser.parse_args()

    # Create a Config instance with the parsed arguments
    config = LLMTransformationConfig(max_turns=None,
                                     max_speakers=None,
                                     cls_task_only=True,
                                     ignore_relation_filter=True,
                                     balance_empty_dialogues=False, 
                                     rebalance_empty_dialogues=False,
                                     rewrite_keys=False,
                                     instruction_type="B",
                                     add_one_shot=False,
                                     )

    # Process and save data
    DataTransformer.process_and_save_data(config)
