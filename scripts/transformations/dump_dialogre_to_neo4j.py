from src.paths import LOCAL_RAW_DATA_PATH
from src.processing.neo4j_operations import Neo4jGraph


if __name__ == "__main__":
    graph = Neo4jGraph()
    dialogues_path = LOCAL_RAW_DATA_PATH / 'dialog-re/data'
    json_files = ["train.json", "test.json", "dev.json"]
    graph.import_dialogre_data(dialogues_path, json_files)
    graph.close()
