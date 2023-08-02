import os
import json
from tqdm import tqdm
from neo4j import GraphDatabase


class Neo4jGraph:
    """
    A class for interacting with Neo4j, primarily used for exporting DialogRE data to Neo4j.
    """

    def __init__(self,
                 uri=os.environ.get('NEO4J_URI'),
                 username=os.environ.get('NEO4J_USERNAME'),
                 password=os.environ.get('NEO4J_PASSWORD')):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

    def close(self):
        self.driver.close()

    def _add_dataset(self, tx, dataset_name):
        tx.run("MERGE (:Dataset {name: $name})", name=dataset_name)

    def _add_dialogue_to_dataset(self, tx, dialogue_id, dataset_name):
        tx.run(
            """
            MATCH (d:Dialogue {id: $dialogue_id})
            MATCH (ds:Dataset {name: $dataset_name})
            MERGE (ds)-[:INCLUDES]->(d)
            """, dialogue_id=dialogue_id, dataset_name=dataset_name
        )

    def import_dialogre_data(self, dialogues_path, json_files):
        with self.driver.session() as session:
            counter = 0
            for json_file in tqdm(json_files):
                dataset_name = json_file.split('.')[0]
                session.write_transaction(self._add_dataset, dataset_name)
                with open(dialogues_path / json_file, 'r', encoding='utf-8') as f:
                    dialogues = json.load(f)
                    for i, dialogue_data in tqdm(enumerate(dialogues)):
                        idx = (i + counter)
                        dialogue, entities_relations = dialogue_data

                        dialogue_exporter = DialogueExporter(session, dialogue, dataset_name, idx)
                        dialogue_exporter.export_dialogue(entities_relations)

                        session.write_transaction(self._add_dialogue_to_dataset, idx, dataset_name)

                counter += (i + 1)

    def archive_and_clean(self, file_path):
        def get_all_data(tx):
            result = tx.run("MATCH (n)-[r]->(m) RETURN n, r, m")
            return result.data()

        with self.driver.session() as session:
            data = session.read_transaction(get_all_data)

        with open(file_path, "w") as f:
            json.dump(data, f)

        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")



class DialogueExporter:
    """
    A class responsible for exporting a single dialogue and its associated data to a Neo4j graph database.
    """

    def __init__(self, session, dialogue_text, dataset_name, dialogue_id=None):
        self.session = session
        self.dialogue_id = dialogue_id
        self.dialogue_text = dialogue_text
        self.dataset_name = dataset_name

    def _add_dialogue(self):
        self.session.run(
            "CREATE (:Dialogue {id: $id, text: $text, dataset: $dataset})", 
            id=self.dialogue_id, text=self.dialogue_text, dataset=self.dataset_name
        )

    def _add_entity(self, entity, entity_type):
        self.session.run("MERGE (:Entity {name: $name, type: $type})", name=entity, type=entity_type)

    def _add_entity_to_dialogue(self, entity):
        self.session.run(
            """
            MATCH (d:Dialogue {id: $dialogue_id})
            MATCH (e:Entity {name: $entity})
            MERGE (d)-[:CONTAINS]->(e)
            """, dialogue_id=self.dialogue_id, entity=entity
        )

    def _add_relation(self, entity1, entity2, relation, trigger):
        #TODO: extend merge to consider entity type
        self.session.run(
            """
            MATCH (a:Entity {name: $entity1})
            MATCH (b:Entity {name: $entity2})
            MERGE (a)-[r:RELATION {type: $relation}]->(b)
            SET r.trigger = coalesce(r.trigger + '; ' + $trigger, $trigger)
            SET r.dialogue_id = coalesce(r.dialogue_id + [$dialogue_id], [$dialogue_id])
            """, 
            dialogue_id=self.dialogue_id, entity1=entity1, entity2=entity2, relation=relation, trigger=trigger
        )

    def _get_max_dialogue_id(self):
        result = self.session.run("MATCH (d:Dialogue) RETURN MAX(d.id) AS max_id")
        max_id = result.single()["max_id"]
        return max_id if max_id is not None else 0

    def export_dialogue(self, entities_relations):
        if self.dialogue_id is None:
            self.dialogue_id = self._get_max_dialogue_id() + 1

        self._add_dialogue()
        for relation in entities_relations:
            x = relation['x']
            y = relation['y']
            x_type = relation['x_type']
            y_type = relation['y_type']

            self._add_entity(x, x_type)
            self._add_entity(y, y_type)

            self._add_entity_to_dialogue(x)
            self._add_entity_to_dialogue(y)

            if relation.get('r_bool', 1) == 1:
                #TODO: fix t and r handling (or remove completely)
                t_values = relation.get('t', [""])
                t = f"{self.dialogue_id}_{t_values[0]}" if t_values != [""] else ""

                r_values = relation.get('r')
                r = ', '.join(r_values) if isinstance(r_values, list) else r_values

                self._add_relation(x, y, r, t)


class DialogueGraphPersister:

    def __init__(self, 
                 dataset_name,
                 uri= os.environ.get('NEO4J_URI') ,
                 username=os.environ.get('NEO4J_USERNAME'),
                 password=os.environ.get('NEO4J_PASSWORD')):
        self.uri = uri
        self.username = username
        self.password = password
        self.dataset_name = dataset_name

        # Initialize the Neo4jGraph
        self.graph = Neo4jGraph(self.uri, self.username, self.password)

    def process_dialogue(self, dialogue, predicted_relations):
        # Open a new Neo4j session
        with self.graph.driver.session() as session:
            # Create the dataset
            session.write_transaction(self.graph._add_dataset, self.dataset_name)
            # Export the dialogue
            exporter = DialogueExporter(session, dialogue, self.dataset_name)
            exporter.export_dialogue(predicted_relations)
            # Link the dialogue to the dataset
            session.write_transaction(self.graph._add_dialogue_to_dataset, exporter.dialogue_id, self.dataset_name)

    def close_connection(self):
        # Close the connection to the Neo4j
        self.graph.close()

    def archive_and_clean(self, file_path):
        self.graph.archive_and_clean(file_path)
