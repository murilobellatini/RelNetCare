import os
import json
from neo4j import GraphDatabase
from src.paths import LOCAL_DATA_PATH
from tqdm import tqdm

class Neo4jGraph:
    def __init__(self, uri, username, password):
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


if __name__ == "__main__":
    uri = "bolt://localhost:7687"  # replace with your Neo4j instance
    username = os.environ.get('NEO4J_USERNAME')  # replace with your username
    password = os.environ.get('NEO4J_PASSWORD')  # replace with your password

    graph = Neo4jGraph(uri, username, password)
    dialogues_path = LOCAL_DATA_PATH / 'dialog-re/data'
    json_files = ["train.json", "test.json", "dev.json"]
    graph.import_dialogre_data(dialogues_path, json_files)
    graph.close()
