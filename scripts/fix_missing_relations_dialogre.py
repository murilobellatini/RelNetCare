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
    for dialogue in data:
        for utterance in dialogue[1]:
            relation_pairs.add((utterance['x'], utterance['y'], utterance['x_type'], utterance['y_type']))
    return relation_pairs


def find_all_relation_combinations(relation_pairs):
    relation_combinations = set()
    for c in itertools.combinations(relation_pairs, 2):
        relation_combinations.add((c[0][0], c[1][1]))
        relation_combinations.add((c[1][0], c[0][1]))
    
    return relation_combinations



def exclude_existing_relations(data, relation_pairs):
    existing_relations = set()
    for dialogue in data:
        for utterance in dialogue[1]:
            existing_relations.add((utterance['x'], utterance['y'], utterance['x_type'], utterance['y_type']))
    
    new_relations = relation_pairs - existing_relations
    return new_relations

def append_new_relations(data, new_relations):
    for dialogue in data:
        for relation_pair in new_relations:
            x, y, x_type, y_type = relation_pair
            new_relation = {
                'y': y,
                'x': x,
                'rid': [38],
                'r': ['no_relation'],
                't': [''],
                'x_type': x_type,
                'y_type': y_type
            }
            dialogue[1].append(new_relation)

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

            # Find relation pairs
            relation_pairs = find_relation_pairs(data)

            all_possible_relations = find_all_relation_combinations(relation_pairs)

            # Exclude existing relations
            new_relations = exclude_existing_relations(data, all_possible_relations)

            # Append new relations
            append_new_relations(data, new_relations)

            # Dump data to new file
            output_file_path = os.path.join(output_folder, filename)
            dump_data(data, output_file_path)

