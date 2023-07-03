from src.processing.etl import DialogRERelationEnricher
from src.paths import LOCAL_PROCESSED_DATA_PATH

if __name__ == "__main__":

    input_dir = LOCAL_PROCESSED_DATA_PATH / "dialog-re-binary"
    output_dir = LOCAL_PROCESSED_DATA_PATH / "dialog-re-binary-enriched-dp"

    relation_enricher = DialogRERelationEnricher()
    relation_enricher.process_dialogues(input_dir, output_dir)
