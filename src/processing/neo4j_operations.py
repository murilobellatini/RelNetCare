import os
import json
from tqdm import tqdm
from neo4j import GraphDatabase
from neo4j_backup import Extractor, Importer


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

    def _purge_database(self, tx):
        tx.run("MATCH (n) DETACH DELETE n;")

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

    def check_and_archive(self, file_path):
        with self.driver.session() as session:
            result = session.run("MATCH (n) RETURN COUNT(n)>0 as nodes_exist")
            nodes_exist = result.single()['nodes_exist']
            if nodes_exist:
                self.archive_and_clean(file_path)
            else:
                print("No nodes exist in the database.")

    def archive_and_clean(self, file_path):
        # Assuming file_path is the path where you want to store the backup
        uri = self.uri
        username = self.username
        password = self.password
        encrypted = False
        trust = "TRUST_ALL_CERTIFICATES"

        with GraphDatabase.driver(uri, auth=(username, password), encrypted=encrypted, trust=trust) as driver:
            database = "neo4j"  # You can change this to the appropriate database name
            project_dir = file_path
            input_yes = False
            compress = True
            indent_size = 4
            json_file_size = int("0xFFFF", 16)

            extractor = Extractor(
                project_dir=project_dir,
                driver=driver,
                database=database,
                input_yes=input_yes,
                compress=compress,
                indent_size=indent_size,
                pull_uniqueness_constraints=True
            )
            extractor.extract_data()

        with self.driver.session() as session:
            session.write_transaction(self._purge_database)

    def load_data(self, file_path):
        # Assuming file_path is the path where you have stored the backup
        uri = self.uri
        username = self.username
        password = self.password
        encrypted = False
        trust = "TRUST_ALL_CERTIFICATES"

        with GraphDatabase.driver(uri, auth=(username, password), encrypted=encrypted, trust=trust) as driver:
            database = "neo4j"  # You can change this to the appropriate database name
            project_dir = file_path
            input_yes = False

            importer = Importer(
                project_dir=project_dir,
                driver=driver,
                database=database,
                input_yes=input_yes
            )
            importer.import_data()
            
    @staticmethod
    def convert_path(path):
        nodes_props = [{'name': node._properties.get('name', None), 'type': node._properties.get('type', None)} for node in path.nodes]
        relationships_props = [{'type': relationship._properties.get('type', None)} for relationship in path.relationships]

        # Collect relationship items
        relations = []
        for i in range(len(relationships_props)):
            relation_item = {
                'x': nodes_props[i]['name'],
                'x_type': nodes_props[i]['type'],
                'y': nodes_props[i+1]['name'],
                'y_type': nodes_props[i+1]['type'],
                'r': relationships_props[i]['type']
            }
            relations.append(relation_item)
        return relations

    def is_graph_empty(self):
        with self.driver.session() as session:
            result = session.run("MATCH (n) RETURN COUNT(n) as node_count")
            node_count = result.single()['node_count']
            return node_count == 0

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

    def check_and_archive(self, file_path):
        self.graph.check_and_archive(file_path)
        
    def load_archived_data(self, archive_subfolder):
        dump_path = os.path.join(archive_subfolder, "neo4j_dump")
        self.graph.load_data(dump_path)
