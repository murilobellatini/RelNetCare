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
from typing import List, Tuple, Optional
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

    
    def __init__(self, raw_data_folder=LOCAL_RAW_DATA_PATH / 'dialog-re/data/'):
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

    def _find_all_relation_combinations(self, relation_pairs):
        relation_combinations = set()
        for c in itertools.combinations(relation_pairs, 2):
            relation_combinations.add((c[0][0], c[1][1]))
            relation_combinations.add((c[1][0], c[0][1]))
        return relation_combinations

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
            for relation_pair in all_new_relations[i]:
                x, x_type = relation_pair[0].split('_')
                y, y_type = relation_pair[1].split('_')
                new_relation = {
                    'y': y,
                    'x': x,
                    'rid': [38],  
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
                if rel['r'][0] == 'no_relation':
                    rel['r'] = ["no_relation"]
                    rel['rid'][0] = 0  # Set 'rid' to 0 for 'no_relation'
                # Check if the relation type is 'unanswerable'
                elif rel['r'][0] == 'unanswerable':
                    rel['r'] = ["unanswerable"]
                    rel['rid'] = [1]  # Set 'rid' to 1 for 'unanswerable'
                else:
                    rel['r'] = ["with_relation"]
                    rel['rid'] = [2]  # Set 'rid' to 2 for 'with_relation'
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
                    all_possible_relations = self._find_all_relation_combinations(relation_pairs)
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

    def tokenize_text(self, text: str, terms: List[str]) -> Tuple[Doc, List[Tuple[int, int]]]:
        """
        Tokenize the text using SpaCy and find the positions of the terms in the tokenized text.

        Args:
        text: str, the text to be tokenized.
        terms: list of str, the terms to find in the text.

        Returns:
        A tuple of two lists:
        - The first list is a list of tokens.
        - The second list is a list of tuples, each tuple represents the span of a term in the tokenized text.
        """
        doc = self.nlp(text)
        tokens = [token.text for token in doc]
        positions = []

        for term in terms:
            term_tokens = term.split()
            term_len = len(term_tokens)

            for i in range(len(tokens) - term_len + 1):
                if tokens[i:i+term_len] == term_tokens:
                    positions.append((i, i+term_len))

        return doc, positions

    def compute_turn_distance(self, dialogue: List[str], relations: List[dict]) -> List[dict]:
        for relation in relations:
            x = relation['x']
            y = relation['y']

            x_turn = [i for i, turn in enumerate(dialogue) if x in turn]
            y_turn = [i for i, turn in enumerate(dialogue) if y in turn]

            if x_turn and y_turn:
                relation["min_turn_distance"] = min([abs(i - j) for i in x_turn for j in y_turn])

        return relations

    def get_dependency_path(self, doc: Doc, x_span: Tuple[int, int], y_span: Tuple[int, int]) -> Optional[List[Span]]:
        # Create the graph
        edges = []
        for token in doc:
            for child in token.children:
                edges.append((token, child))
        graph = nx.Graph(edges)

        # Get the first token of x and the last token of y
        x_token = doc[x_span[0]]
        y_token = doc[y_span[1] - 1]

        # Compute the shortest path
        try:
            shortest_path = nx.shortest_path(graph, source=x_token, target=y_token)
            print('Path computed successfully!')
        except Exception as e:
            print(e)
            shortest_path = None
        
        return shortest_path

    def compute_distance(self, dialogue: List[str], relations: List[dict]) -> List[dict]:
        dialogue_str = ' '.join(dialogue)

        for relation in relations:
            x = relation['x']
            y = relation['y']

            doc, x_positions = self.tokenize_text(dialogue_str, [x])
            _, y_positions = self.tokenize_text(dialogue_str, [y])

            if x_positions and y_positions:
                min_distance = min([abs(x[1] - y[0]) for x in x_positions for y in y_positions])

                x_min = min([x[0] for x in x_positions])
                y_min = min([y[0] for y in y_positions])
                x_max = max([x[1] for x in x_positions])
                y_max = max([y[1] for y in y_positions])

                if x_min < y_min:
                    relation["x_span"] = (x_min, x_max)
                    relation["y_span"] = (y_min, y_max)
                else:
                    relation["x_span"] = (y_min, y_max)
                    relation["y_span"] = (x_min, x_max)

                relation["min_words_distance"] = min_distance
                
                relation['dependency_path'] = self.get_dependency_path(doc, relation["x_span"], relation["y_span"])

        relations = self.compute_turn_distance(dialogue, relations)

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
                        relations_with_distances = self.compute_distance(dialogue, relations)
                        processed_dialogues.append((dialogue, relations_with_distances))

                    with open(output_file, 'w', encoding='utf8') as f:
                        json.dump(processed_dialogues, f)
