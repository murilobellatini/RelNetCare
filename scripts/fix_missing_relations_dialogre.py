import pandas as pd
import copy
import json
import os
import itertools

from src.paths import LOCAL_RAW_DATA_PATH, LOCAL_PROCESSED_DATA_PATH

def load_data(file_path):
    with open(file_path, 'r', encoding='utf8') as file:
        data = json.load(file)
    return data

def find_relation_pairs(data):
    relation_pairs = set()
    for r in data[1]:  # This gives you a list of relations
        x = f"{r['x']}_{r['x_type']}"
        y = f"{r['y']}_{r['y_type']}"
        relation_pairs.add((x, y))
    return relation_pairs

def find_all_relation_combinations(relation_pairs):
    relation_combinations = set()
    for c in itertools.combinations(relation_pairs, 2):
        relation_combinations.add((c[0][0], c[1][1]))
        relation_combinations.add((c[1][0], c[0][1]))
    return relation_combinations

def exclude_existing_relations(data, relation_pairs):
    existing_relations = set()
    for relation in data[1]:
        x = f"{relation['x']}_{relation['x_type']}"
        y = f"{relation['y']}_{relation['y_type']}"
        existing_relations.add((x, y))
    new_relations = relation_pairs - existing_relations
    return new_relations

def create_new_dialogues_with_new_relations(data, all_new_relations):
    new_dialogues = []
    for i, dialogue in enumerate(data):
        new_dialogue = copy.deepcopy(dialogue)  # Deep copy to avoid modifying original data
        for relation_pair in all_new_relations[i]:
            x, x_type = relation_pair[0].split('_')
            y, y_type = relation_pair[1].split('_')
            new_relation = {
                'y': y,
                'x': x,
                'rid': [38],  # Replace with correct id
                'r': ['no_relation'],
                't': [''],
                'x_type': x_type,
                'y_type': y_type
            }
            new_dialogue[1].append(new_relation)
        new_dialogues.append(new_dialogue)
    return new_dialogues

def dump_data(data, file_path):
    with open(file_path, 'w', encoding='utf8') as file:
        json.dump(data, file)

if __name__ == "__main__":
    # Set the file paths
    input_folder  = LOCAL_RAW_DATA_PATH / 'dialog-re/data/'
    output_folder = LOCAL_PROCESSED_DATA_PATH / 'dialog-re-fixed-relations'
    os.makedirs(output_folder, exist_ok=True)
    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            # Load the data
            input_file_path = os.path.join(input_folder, filename)
            data = load_data(input_file_path)
            # Initialize a list to store all the new relations for each dialogue in the file
            all_new_relations = []
            for dialogue in data:
                # Find relation pairs
                relation_pairs = find_relation_pairs(dialogue)
                all_possible_relations = find_all_relation_combinations(relation_pairs)
                # Exclude existing relations
                new_relations = exclude_existing_relations(dialogue, all_possible_relations)
                # Append new relations to the list
                all_new_relations.append(new_relations)

            # Create new dialogues with new relations
            new_data = create_new_dialogues_with_new_relations(data, all_new_relations)
            # Dump data to new file
            output_file_path = os.path.join(output_folder, filename)


            df = pd.DataFrame(data)
            relation_counts =  df[1].apply(lambda x: x[0]['r'][0]).value_counts()
            print('relation_counts=',relation_counts)

            n_df = pd.DataFrame(new_data)
            n_relation_counts = n_df[1].apply(lambda x: x[0]['r'][0]).value_counts()
            print('nrelation_counts=',n_relation_counts)

            dump_data(new_data, output_file_path)
