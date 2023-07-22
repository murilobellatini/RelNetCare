from src.processing.dialogre_processing import DialogRERelationEnricher
from src.paths import LOCAL_PROCESSED_DATA_PATH



if __name__ == "__main__":

    input_dir = LOCAL_PROCESSED_DATA_PATH / "dialog-re-binary-validated"
    output_dir = LOCAL_PROCESSED_DATA_PATH / "dialog-re-binary-validated-enriched"
    
    enricher = DialogRERelationEnricher()

    enricher.process_dialogues(input_dir, output_dir)
