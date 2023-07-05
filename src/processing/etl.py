import os
import glob
import json
import copy
import shutil
import itertools
import nltk
import spacy
import pandas as pd
from tqdm import tqdm
import networkx as nx
from pathlib import Path
from collections import Counter
from neo4j import GraphDatabase
from spacy.tokens import Span, Doc
from typing import List, Tuple, Optional, Text, Dict
from sklearn.utils import resample
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler



from src.paths import LOCAL_RAW_DATA_PATH, LOCAL_PROCESSED_DATA_PATH

class Neo4jGraph:
    """
    A class for interacting with Neo4j, primarily used for exporting DialogRE data to Neo4j.
    """

    def __init__(self,
                 uri= "bolt://localhost:7687" ,
                 username=os.environ.get('NEO4J_USERNAME'),
                 password=os.environ.get('NEO4J_PASSWORD')):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

    def close(self):
        self.driver.close()

    def _add_dialogue(self, tx, dialogue_id, dialogue_text, dataset):
        tx.run("CREATE (:Dialogue {id: $id, text: $text, dataset: $dataset})", id=dialogue_id, text=dialogue_text,
               dataset=dataset)

    def _add_entity(self, tx, entity, entity_type):
        tx.run("MERGE (:Entity {name: $name, type: $type})", name=entity, type=entity_type)

    def _add_entity_to_dialogue(self, tx, dialogue_id, entity):
        tx.run("""
            MATCH (d:Dialogue {id: $dialogue_id})
            MATCH (e:Entity {name: $entity})
            MERGE (d)-[:CONTAINS]->(e)
            """, dialogue_id=dialogue_id, entity=entity)

    def _add_relation(self, tx, dialogue_id, entity1, entity2, relation, trigger):
        tx.run("""
            MATCH (a:Entity {name: $entity1})
            MATCH (b:Entity {name: $entity2})
            MERGE (a)-[r:RELATION {type: $relation}]->(b)
            SET r.trigger = coalesce(r.trigger + '; ' + $trigger, $trigger)
            SET r.dialogue_id = coalesce(r.dialogue_id + [$dialogue_id], [$dialogue_id])
            """, dialogue_id=dialogue_id, entity1=entity1, entity2=entity2, relation=relation, trigger=trigger)

    def _add_dataset(self, tx, dataset_name):
        tx.run("CREATE (:Dataset {name: $name})", name=dataset_name)

    def _add_dialogue_to_dataset(self, tx, dialogue_id, dataset_name):
        tx.run("""
            MATCH (d:Dialogue {id: $dialogue_id})
            MATCH (ds:Dataset {name: $dataset_name})
            MERGE (ds)-[:INCLUDES]->(d)
            """, dialogue_id=dialogue_id, dataset_name=dataset_name)

    def import_dialogre_data(self, dialogues_path, json_files):
        with self.driver.session() as session:
            counter = 0
            for json_file in tqdm(json_files):
                dataset_name = json_file.split('.')[0]  # assuming the dataset name is the filename without extension
                session.execute_write(self._add_dataset, dataset_name)  # create a dataset node
                with open(dialogues_path / json_file, 'r', encoding='utf-8') as f:
                    dialogues = json.load(f)
                    for i, dialogue_data in tqdm(enumerate(dialogues)):
                        idx = (i + counter)
                        dialogue, entities_relations = dialogue_data

                        # Add the dialogue to the graph and associate it with the dataset
                        session.execute_write(self._add_dialogue, idx, dialogue, dataset_name)
                        session.execute_write(self._add_dialogue_to_dataset, idx, dataset_name)

                        for relation in entities_relations:
                            x = relation['x']
                            y = relation['y']
                            r = ', '.join(relation['r'])
                            t = f"{idx}_{relation['t'][0]}" if relation['t'] != [""] else "" # Prepend the trigger with the current idx

                            if "Speaker" in x:
                                x = f"{idx}_{x}"
                            if "Speaker" in y:
                                y = f"{idx}_{y}"

                            x_type = relation['x_type']
                            y_type = relation['y_type']

                            # Add the entities to the graph
                            session.execute_write(self._add_entity, x, x_type)
                            session.execute_write(self._add_entity, y, y_type)

                            # Associate the entities with the dialogue
                            session.execute_write(self._add_entity_to_dialogue, idx, x)
                            session.execute_write(self._add_entity_to_dialogue, idx, y)

                            # Add the relationship to the graph
                            session.execute_write(self._add_relation, idx, x, y, r, t)

                counter += (i + 1)


class DialogREDatasetTransformer:
    """
    A utility for modifying DialogRE datasets, emphasizing cases without relations. Key methods:

    `add_no_relation_labels`: Adds "no relation" instances to the dataset.

    `transform_to_binary`: Transforms the dataset to express "no relation", "unanswerable", or "with relation".
    
    `transform_to_ternary`: Transforms the dataset into a binary version, merging "unanswerable" with "no relation",
                            and renaming all other relations as "with relation".
    """

    
    def __init__(self, raw_data_folder=LOCAL_RAW_DATA_PATH / 'dialog-re/'):
        self.raw_data_folder = raw_data_folder
        self.df = pd.DataFrame(columns=["Dialogue", "Relations", "Origin"])

    def load_data_to_dataframe(self):
        # Get a list of all json files in the directory, excluding 'relation_label_dict'
        files = [Path(f) for f in glob.glob(f"{self.raw_data_folder}/*.json") if "relation_label_dict" not in str(f)]

        # Loop over all json files in the directory
        for file_name in files:
            with open(file_name, 'r') as file:
                data = json.load(file)

                # Convert the data to a DataFrame
                df_temp = pd.DataFrame(data, columns=["Dialogue", "Relations"])

                # Add a new column to this DataFrame for the origin
                df_temp["Origin"] = file_name.stem  # This will get just the file name without the extension

                # Append the temporary DataFrame to the main DataFrame
                self.df = pd.concat([self.df, df_temp], ignore_index=True)

        return self.df

    def _load_data(self, file_path):
        with open(file_path, 'r', encoding='utf8') as file:
            data = json.load(file)
        return data

    def _find_relation_pairs(self, data):
        relation_pairs = set()
        for r in data[1]:  
            x = f"{r['x']}_{r['x_type']}"
            y = f"{r['y']}_{r['y_type']}"
            relation_pairs.add((x, y))
        return relation_pairs

    def _find_all_relation_permutations(self, relation_pairs):
        unique_items = set(item for sublist in relation_pairs for item in sublist)
        unique_combinations = set(itertools.permutations(unique_items, 2))
        return unique_combinations

    def _exclude_existing_relations(self, data, relation_pairs):
        existing_relations = set()
        for relation in data[1]:
            x = f"{relation['x']}_{relation['x_type']}"
            y = f"{relation['y']}_{relation['y_type']}"
            existing_relations.add((x, y))
        new_relations = relation_pairs - existing_relations
        return new_relations

    def _create_new_dialogues_with_new_relations(self, data, all_new_relations):
        new_dialogues = []
        for i, dialogue in enumerate(data):
            new_dialogue = copy.deepcopy(dialogue)
            relation_pairs = self._find_relation_pairs(dialogue)  # get the existing relation pairs
            for relation_pair in all_new_relations[i]:
                x, x_type = relation_pair[0].split('_')
                y, y_type = relation_pair[1].split('_')
                inverse_pair = (f"{y}_{y_type}", f"{x}_{x_type}")
                if inverse_pair in relation_pairs:  # check if the inverse pair exists
                    new_relation = {
                        'y': y,
                        'x': x,
                        'rid': [39],
                        'r': ['inverse_relation'],
                        't': [''],
                        'x_type': x_type,
                        'y_type': y_type
                    }
                else:
                    new_relation = {
                        'y': y,
                        'x': x,
                        'rid': [38],  # set to 38 for 'no_relation'
                        'r': ['no_relation'],
                        't': [''],
                        'x_type': x_type,
                        'y_type': y_type
                    }
                new_dialogue[1].append(new_relation)
            new_dialogues.append(new_dialogue)
        return new_dialogues

    def _dump_data(self, data, file_path):
        os.makedirs(Path(file_path).parents[0], exist_ok=True)
        with open(file_path, 'w', encoding='utf8') as file:
            json.dump(data, file)

    def _dump_relation_label_dict(self, data, output_path):
        # Flatten the data into a list of dictionaries
        flat_data = [item for sublist in data for item in sublist[1]]

        # Create a dataframe from the flattened data
        df = pd.DataFrame(flat_data)

        # Take the first element of each list in 'rid' and 'r' columns
        df['rid'] = df['rid'].apply(lambda x: x[0])
        df['r'] = df['r'].apply(lambda x: x[0])

        # Extract unique (rid, r) pairs, and convert the DataFrame to a dictionary
        label_dict = df[['rid', 'r']].drop_duplicates().set_index('rid').to_dict()['r']

        # Sort the dictionary by keys
        sorted_label_dict = {k: label_dict[k] for k in sorted(label_dict)}

        # Save the label dictionary to json file
        with open(output_path, 'w') as file:
            json.dump(sorted_label_dict, file)

        print(f"Label dictionary saved to {output_path}")

    def _overwrite_relations(self, data):
        for item in data:
            # item[1] corresponds to the list of relations
            for rel in item[1]:
                # Check if the relation type is 'no_relation'
                if rel['r'][0] == 'no_relation' or rel['r'][0] == 'unanswerable':
                    rel['r'] = ["no_relation"]
                    rel['rid'][0] = 0  # Set 'rid' to 0 for 'no_relation'
                # Check if the relation type is 'unanswerable'
                elif rel['r'][0] == 'inverse_relation':
                    rel['r'] = ["inverse_relation"]
                    rel['rid'] = [2]  # Set 'rid' to 1 for 'unanswerable'
                else:
                    rel['r'] = ["with_relation"]
                    rel['rid'] = [1]  # Set 'rid' to 2 for 'with_relation'
        return data

    def _merge_unanswerable_and_no_relation(self, data):
        for item in data:
            for rel in item[1]:
                # Check if the relation type is 'no_relation' or 'unanswerable'
                if rel['r'][0] == 'no_relation' or rel['r'][0] == 'unanswerable':
                    rel['r'] = ["no_relation_unanswerable" ]
                    rel['rid'] = [0]  # Set 'rid' to 0 for 'no_relation' and 'unanswerable'
                else:
                    rel['r'] = ["with_relation"]
                    rel['rid'] = [1]  # Set 'rid' to 1 for 'with_relation'
        return data

    def transform_to_binary(self,
                            input_folder=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-with-no-relation',
                            output_folder=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-binary'):
        
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"The folder '{input_folder}' does not exist. Please run method `add_no_relation_labels` first.")

        os.makedirs(output_folder, exist_ok=True)
        files = [Path(f) for f in glob.glob(str(input_folder / "*.json")) if 'relation_label_dict.json' not in str(f)]

        for file in files:
            with open(file, 'r') as json_file:
                data = json.load(json_file)

            # Merge 'unanswerable' and 'no_relation', and rename all other relations to 'with_relation'
            data = self._merge_unanswerable_and_no_relation(data)

            # Determine the set (train, dev, test) based on the filename
            set_type = file.stem.split('_')[-1]  # This assumes that the set type is always at the end of the file name

            # Write back to a new JSON file
            with open(output_folder / f"{set_type}.json", 'w') as json_file:
                json.dump(data, json_file)

        # Dump the new label dictionary
        self._dump_relation_label_dict(data, output_folder / 'relation_label_dict.json')

    def transform_to_ternary(self,
                             input_folder=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-with-no-relation',
                             output_folder=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-ternary'):

        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"The folder '{input_folder}' does not exist. Please run method `add_no_relation_labels` first.")

        os.makedirs(output_folder, exist_ok=True)
        files = [Path(f) for f in glob.glob(str(input_folder / "*.json")) if 'relation_label_dict.json' not in str(f)]

        for file in files:
            with open(file, 'r') as json_file:
                data = json.load(json_file)

            # Overwrite relations
            data = self._overwrite_relations(data)

            # Determine the set (train, dev, test) based on the filename
            set_type = file.stem.split('_')[-1]  # This assumes that the set type is always at the end of the file name

            # Write back to a new JSON file
            with open(output_folder / f"{set_type}.json", 'w') as json_file:
                json.dump(data, json_file)

        # Dump the new label dictionary
        self._dump_relation_label_dict(data, output_folder / 'relation_label_dict.json')

    def add_no_relation_labels(self,
                               output_folder=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-with-no-relation'):

        os.makedirs(output_folder, exist_ok=True)
        for filename in os.listdir(self.raw_data_folder):
            if 'relation_label_dict' in filename:
                continue
            if filename.endswith('.json'):
                input_file_path = os.path.join(self.raw_data_folder, filename)
                data = self._load_data(input_file_path)
                all_new_relations = []
                for dialogue in data:
                    relation_pairs = self._find_relation_pairs(dialogue)
                    all_possible_relations = self._find_all_relation_permutations(relation_pairs)
                    new_relations = self._exclude_existing_relations(dialogue, all_possible_relations)
                    all_new_relations.append(new_relations)

                new_data = self._create_new_dialogues_with_new_relations(data, all_new_relations)

                output_file_path = os.path.join(output_folder, filename)
                self._dump_data(new_data, output_file_path)

        # Dump the new label dictionary
        self._dump_relation_label_dict(new_data, output_folder / 'relation_label_dict.json')


class DialogREDatasetBalancer(DialogREDatasetTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _filter_dialogues(self, data):
        all_relations = set()
        for dialogue in data:
            all_relations.update(relation['r'][0] for relation in dialogue[1])

        filtered_data = [dialogue for dialogue in data if all_relations.issubset(relation['r'][0] for relation in dialogue[1])]
        
        print(f"Original dialogue count: {len(data)}, Filtered dialogue count: {len(filtered_data)}")
        print(f"Original relations count: {sum(len(dialogue[1]) for dialogue in data)}, Filtered relations count: {sum(len(dialogue[1]) for dialogue in filtered_data)}")

        return filtered_data

    def _resample_dialogue(self, dialogue, sampler):
        X = [[i] for i in range(len(dialogue[1]))]
        y = [relation['r'][0] for relation in dialogue[1]]

        X_res, _ = sampler.fit_resample(X, y)
        resampled_relations = [dialogue[1][i[0]] for i in X_res]

        return [dialogue[0], resampled_relations]

    def _resample(self, data, sampler):
        resampled_data = [self._resample_dialogue(dialogue, sampler) for dialogue in data]

        return resampled_data
    
    def _copy_other_files(self, input_folder, output_folder, ignore_files=None):
        """
        Copy all files from the input folder to the output folder,
        excluding the ones specified in the ignore_files list.
        """
        for filename in os.listdir(input_folder):
            if filename in ignore_files:
                continue

            shutil.copy(os.path.join(input_folder, filename),
                        os.path.join(output_folder, filename)) 

    def undersample(self, train_file, output_folder):
        data = self._load_data(train_file)
        filtered_data = self._filter_dialogues(data)
        
        undersampler = RandomUnderSampler(random_state=42)
        resampled_data = self._resample(filtered_data, undersampler)

        output_file_path = os.path.join(output_folder, train_file.name)
        self._dump_data(resampled_data, output_file_path)
        self._copy_other_files(train_file.parents[0], output_folder, ignore_files=['train.json'])


    def oversample(self, train_file, output_folder):
        data = self._load_data(train_file)
        filtered_data = self._filter_dialogues(data)

        oversampler = RandomOverSampler(random_state=42)
        resampled_data = self._resample(filtered_data, oversampler)

        output_file_path = os.path.join(output_folder, train_file.name)
        self._dump_data(resampled_data, output_file_path)
        self._copy_other_files(train_file.parents[0], output_folder, ignore_files=['train.json'])



class DialogRERelationEnricher:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        nltk.download('punkt')

    def _tokenize_text(self, text: str, terms: List[str]) -> Tuple[Doc, Dict[str, List[Tuple[int, int]]], Dict[str, List[Tuple[int, int]]]]:
        doc = self.nlp(text)
        tokens = [token.text for token in doc]
        token_positions = {term: [] for term in terms}
        char_positions = {term: [] for term in terms}

        for term in terms:
            term_tokens = term.split()
            term_len = len(term_tokens)

            for i in range(len(tokens) - term_len + 1):
                if tokens[i:i+term_len] == term_tokens:
                    token_positions[term].append((i, i+term_len))
                    # Convert token span to char span
                    char_positions[term].append((doc[i].idx, doc[i+term_len-1].idx + len(doc[i+term_len-1])))

        return doc, token_positions, char_positions

    def _get_entities(self, relations: List[dict]) -> List[str]:
        entities = set()
        for relation in relations:
            entities.add(relation['x'])
            entities.add(relation['y'])
        return list(entities)

    def _compute_turn_distance(self, dialogue: List[str], relations: List[dict]) -> List[dict]:
        for relation in relations:
            x = relation['x']
            y = relation['y']

            x_turn = [i for i, turn in enumerate(dialogue) if x in turn]
            y_turn = [i for i, turn in enumerate(dialogue) if y in turn]

            if x_turn and y_turn:
                relation["min_turn_distance"] = min([abs(i - j) for i in x_turn for j in y_turn])
                relation["min_turn_distance_pct"] = relation["min_turn_distance"] / len('\n'.join(dialogue))
            
        return relations

    def _get_spacy_features(self, doc: Doc, x_span: Tuple[int, int], y_span: Tuple[int, int]) -> Optional[List[Text]]:

        # Get the root token of x and y
        x_token = doc[x_span[0]]
        y_token = doc[y_span[1] - 1]
        
        return {
            "x_pos": x_token.pos_,
            "x_dep": x_token.dep_,
            "x_tag": x_token.tag_,
            "y_pos": y_token.pos_,
            "y_dep": y_token.dep_,
            "y_tag": y_token.tag_,
            }
        
    def _get_connecting_text(self, doc: Doc, x_span: Tuple[int, int], y_span: Tuple[int, int]) -> Optional[List[Text]]:
        # Get the root token of x and y
        x_token = doc[x_span[0]]
        y_token = doc[y_span[1] - 1]
        
        return doc[x_span[0]:y_span[1] + len(y_token.text)].text

    def _get_dependency_path(self, doc: Doc, x_span: Tuple[int, int], y_span: Tuple[int, int]) -> Optional[List[Text]]:
        # Create the graph
        edges = [(token, child) for token in doc for child in token.children]
        graph = nx.Graph(edges)

        # Get the root token of x and y
        x_token = doc[x_span[0]]
        y_token = doc[y_span[1] - 1]

        # Compute the shortest path
        try:
            shortest_path = [t.text for t in nx.shortest_path(graph, source=x_token, target=y_token)]
            print('Path computed successfully!')
        except nx.NodeNotFound as e:
            print(f'Node not found: {e}')
            shortest_path = []
        except nx.NetworkXNoPath as e:
            print(f'No path between nodes: {e}')
            shortest_path = []
        except Exception as e:
            print(f'An unexpected error occurred: {e}')
            shortest_path = []
            
        return shortest_path


    def _compute_distance(self, dialogue: List[str], relations: List[dict]) -> List[dict]:
        dialogue_str = ' '.join(dialogue)

        # Get all unique entities from relations
        entities = self._get_entities(relations)

        # Compute token and char positions for each entity
        doc, entity_token_positions, entity_char_positions = self._tokenize_text(dialogue_str, entities)

        for relation in relations:
            x = relation['x']
            y = relation['y']

            # Get pre-computed token and char positions
            x_token_positions = entity_token_positions[x]
            y_token_positions = entity_token_positions[y]
            x_char_positions = entity_char_positions[x]
            y_char_positions = entity_char_positions[y]
            
            if x_token_positions and y_token_positions:
                min_distance = min([abs(x[1] - y[0]) for x in x_token_positions for y in y_token_positions])

                x_min_token = min([x[0] for x in x_token_positions])
                y_min_token = min([y[0] for y in y_token_positions])
                x_max_token = max([x[1] for x in x_token_positions])
                y_max_token = max([y[1] for y in y_token_positions])

                if x_min_token < y_min_token:
                    relation["x_token_span"] = (x_min_token, x_max_token)
                    relation["y_token_span"] = (y_min_token, y_max_token)
                else:
                    relation["x_token_span"] = (y_min_token, y_max_token)
                    relation["y_token_span"] = (x_min_token, x_max_token)

                x_min_char = min([x[0] for x in x_char_positions])
                y_min_char = min([y[0] for y in y_char_positions])
                x_max_char = max([x[1] for x in x_char_positions])
                y_max_char = max([y[1] for y in y_char_positions])

                if x_min_char < y_min_char:
                    relation["x_char_span"] = (x_min_char, x_max_char)
                    relation["y_char_span"] = (y_min_char, y_max_char)
                else:
                    relation["x_char_span"] = (y_min_char, y_max_char)
                    relation["y_char_span"] = (x_min_char, x_max_char)

                relation["min_words_distance"] = min_distance
                relation["min_words_distance_pct"] = min_distance / len(dialogue_str)
                
                relation['spacy_features'] = self._get_spacy_features(doc, relation["x_token_span"], relation["y_token_span"])
                # relation['connecting_text'] = self._get_connecting_text(doc, relation["x_token_span"], relation["y_token_span"])
                # relation['dependency_path'] = self._get_dependency_path(doc, relation["x_token_span"], relation["y_token_span"])

        relations = self._compute_turn_distance(dialogue, relations)

        return relations

    def process_dialogues(self, input_dir: str, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for filename in os.listdir(input_dir):
            print("Processing file: {}".format(filename))
            if filename.endswith(".json"):
                input_file = os.path.join(input_dir, filename)
                output_file = os.path.join(output_dir, filename)

                if filename == "relation_label_dict.json":
                    shutil.copyfile(input_file, output_file)
                else:
                    with open(input_file, 'r', encoding='utf8') as f:
                        dialogues_relations = json.load(f)

                    processed_dialogues = []
                    for dialogue, relations in tqdm(dialogues_relations):
                        relations_with_distances = self._compute_distance(dialogue, relations)
                        processed_dialogues.append((dialogue, relations_with_distances))

                    # Assert that 'data' and 'new_data' have the same length
                    assert len(dialogues_relations) == len(processed_dialogues), "Data and new data have different lengths"

                    # Assert that 'data' and 'new_data' have the same relation count for each dialogue
                    for original_dialogue, new_dialogue in zip(dialogues_relations, processed_dialogues):
                        assert len(original_dialogue[1]) == len(new_dialogue[1]), "Original and new dialogues have different relation counts"

                    # Filter relations to only include those with a 'min_words_distance' key
                    for i, (dialogue, relations) in enumerate(processed_dialogues):
                        processed_dialogues[i] = (dialogue, [r for r in relations if 'min_words_distance' in r])

                    # Filter dialogues to only include those with at least one relation containing a 'min_words_distance' key
                    processed_dialogues = [(dialogue, relations) for (dialogue, relations) in processed_dialogues if any('min_words_distance' in r for r in relations)]

                    # Check that every new relation contains a 'min_words_distance' key
                    for _, relations in processed_dialogues:
                        for r in relations:
                            assert 'min_words_distance' in r, "Item in new relation does not contain 'min_words_distance' key"
                            
                    with open(output_file, 'w', encoding='utf8') as f:
                        json.dump(processed_dialogues, f)
