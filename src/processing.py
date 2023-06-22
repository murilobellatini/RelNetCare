import os
import json
from tqdm import tqdm
from neo4j import GraphDatabase

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

    def add_dialogue(self, tx, dialogue_id, dialogue_text, dataset):
        tx.run("CREATE (:Dialogue {id: $id, text: $text, dataset: $dataset})", id=dialogue_id, text=dialogue_text,
               dataset=dataset)

    def add_entity(self, tx, entity, entity_type):
        tx.run("MERGE (:Entity {name: $name, type: $type})", name=entity, type=entity_type)

    def add_entity_to_dialogue(self, tx, dialogue_id, entity):
        tx.run("""
            MATCH (d:Dialogue {id: $dialogue_id})
            MATCH (e:Entity {name: $entity})
            MERGE (d)-[:CONTAINS]->(e)
            """, dialogue_id=dialogue_id, entity=entity)

    def add_relation(self, tx, dialogue_id, entity1, entity2, relation, trigger):
        tx.run("""
            MATCH (a:Entity {name: $entity1})
            MATCH (b:Entity {name: $entity2})
            MERGE (a)-[r:RELATION {type: $relation}]->(b)
            SET r.trigger = coalesce(r.trigger + '; ' + $trigger, $trigger)
            SET r.dialogue_id = coalesce(r.dialogue_id + [$dialogue_id], [$dialogue_id])
            """, dialogue_id=dialogue_id, entity1=entity1, entity2=entity2, relation=relation, trigger=trigger)

    def add_dataset(self, tx, dataset_name):
        tx.run("CREATE (:Dataset {name: $name})", name=dataset_name)

    def add_dialogue_to_dataset(self, tx, dialogue_id, dataset_name):
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
                session.execute_write(self.add_dataset, dataset_name)  # create a dataset node
                with open(dialogues_path / json_file, 'r', encoding='utf-8') as f:
                    dialogues = json.load(f)
                    for i, dialogue_data in tqdm(enumerate(dialogues)):
                        idx = (i + counter)
                        dialogue, entities_relations = dialogue_data

                        # Add the dialogue to the graph and associate it with the dataset
                        session.execute_write(self.add_dialogue, idx, dialogue, dataset_name)
                        session.execute_write(self.add_dialogue_to_dataset, idx, dataset_name)

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
                            session.execute_write(self.add_entity, x, x_type)
                            session.execute_write(self.add_entity, y, y_type)

                            # Associate the entities with the dialogue
                            session.execute_write(self.add_entity_to_dialogue, idx, x)
                            session.execute_write(self.add_entity_to_dialogue, idx, y)

                            # Add the relationship to the graph
                            session.execute_write(self.add_relation, idx, x, y, r, t)

                counter += (i + 1)

import pandas as pd
import copy
import json
import os
import itertools

from src.paths import LOCAL_RAW_DATA_PATH, LOCAL_PROCESSED_DATA_PATH

class DialogREDatasetFixer:
    """
    A class for processing DialogRE datasets by adding the 'no_relation' relation, 
    aiding in predicting whether a relationship exists between entities.
    """

    def __init__(self, input_folder=LOCAL_RAW_DATA_PATH / 'dialog-re/data/', output_folder=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-fixed-relations'):
        self.input_folder = input_folder
        self.output_folder = output_folder

    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf8') as file:
            data = json.load(file)
        return data

    def find_relation_pairs(self, data):
        relation_pairs = set()
        for r in data[1]:  
            x = f"{r['x']}_{r['x_type']}"
            y = f"{r['y']}_{r['y_type']}"
            relation_pairs.add((x, y))
        return relation_pairs

    def find_all_relation_combinations(self, relation_pairs):
        relation_combinations = set()
        for c in itertools.combinations(relation_pairs, 2):
            relation_combinations.add((c[0][0], c[1][1]))
            relation_combinations.add((c[1][0], c[0][1]))
        return relation_combinations

    def exclude_existing_relations(self, data, relation_pairs):
        existing_relations = set()
        for relation in data[1]:
            x = f"{relation['x']}_{relation['x_type']}"
            y = f"{relation['y']}_{relation['y_type']}"
            existing_relations.add((x, y))
        new_relations = relation_pairs - existing_relations
        return new_relations

    def create_new_dialogues_with_new_relations(self, data, all_new_relations):
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

    def dump_data(self, data, file_path):
        with open(file_path, 'w', encoding='utf8') as file:
            json.dump(data, file)

    def process(self):
        os.makedirs(self.output_folder, exist_ok=True)
        for filename in os.listdir(self.input_folder):
            if filename.endswith('.json'):
                input_file_path = os.path.join(self.input_folder, filename)
                data = self.load_data(input_file_path)
                all_new_relations = []
                for dialogue in data:
                    relation_pairs = self.find_relation_pairs(dialogue)
                    all_possible_relations = self.find_all_relation_combinations(relation_pairs)
                    new_relations = self.exclude_existing_relations(dialogue, all_possible_relations)
                    all_new_relations.append(new_relations)

                new_data = self.create_new_dialogues_with_new_relations(data, all_new_relations)

                output_file_path = os.path.join(self.output_folder, filename)
                self.dump_data(new_data, output_file_path)
