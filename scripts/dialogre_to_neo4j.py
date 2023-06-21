import os
import json
from neo4j import GraphDatabase
from src.paths import LOCAL_DATA_PATH
from tqdm import tqdm

uri = "bolt://localhost:7687"  # replace with your Neo4j instance
driver = GraphDatabase.driver(uri, auth=(os.environ.get('NEO4J_USERNAME'), os.environ.get('NEO4J_PASSWORD')))  # replace with your username and password

# Define the path to your json files
dialogues_path = LOCAL_DATA_PATH / 'dialog-re/data'
json_files = ["train.json", "test.json", "dev.json"]

def add_dialogue(tx, dialogue_id, dialogue_text, dataset):
    tx.run("CREATE (:Dialogue {id: $id, text: $text, dataset: $dataset})", id=dialogue_id, text=dialogue_text, dataset=dataset)

def add_entity(tx, entity):
    tx.run("MERGE (:Entity {name: $name})", name=entity)

def add_entity_to_dialogue(tx, dialogue_id, entity):
    tx.run("""
        MATCH (d:Dialogue {id: $dialogue_id})
        MATCH (e:Entity {name: $entity})
        MERGE (d)-[:CONTAINS]->(e)
        """, dialogue_id=dialogue_id, entity=entity)

def add_relation(tx, entity1, entity2, relation):
    tx.run("""
        MATCH (a:Entity {name: $entity1})
        MATCH (b:Entity {name: $entity2})
        MERGE (a)-[:RELATION {type: $relation}]->(b)
        """, entity1=entity1, entity2=entity2, relation=relation)

def add_dataset(tx, dataset_name):
    tx.run("CREATE (:Dataset {name: $name})", name=dataset_name)

def add_dialogue_to_dataset(tx, dialogue_id, dataset_name):
    tx.run("""
        MATCH (d:Dialogue {id: $dialogue_id})
        MATCH (ds:Dataset {name: $dataset_name})
        MERGE (ds)-[:INCLUDES]->(d)
        """, dialogue_id=dialogue_id, dataset_name=dataset_name)


with driver.session() as session:
    for json_file in tqdm(json_files):
        dataset_name = json_file.split('.')[0]  # assuming the dataset name is the filename without extension
        session.write_transaction(add_dataset, dataset_name)  # create a dataset node
        with open(dialogues_path / json_file, 'r', encoding='utf-8') as f:
            dialogues = json.load(f)
            for i, dialogue_data in tqdm(enumerate(dialogues)):
                dialogue, entities_relations = dialogue_data

                # Add the dialogue to the graph and associate it with the dataset
                session.write_transaction(add_dialogue, i, dialogue, dataset_name)
                session.write_transaction(add_dialogue_to_dataset, i, dataset_name)

                for relation in entities_relations:
                    x = relation['x']
                    y = relation['y']
                    r = ', '.join(relation['r'])

                    # Add the entities to the graph
                    session.write_transaction(add_entity, x)
                    session.write_transaction(add_entity, y)

                    # Associate the entities with the dialogue
                    session.write_transaction(add_entity_to_dialogue, i, x)
                    session.write_transaction(add_entity_to_dialogue, i, y)

                    # Add the relationship to the graph
                    session.write_transaction(add_relation, x, y, r)

driver.close()
