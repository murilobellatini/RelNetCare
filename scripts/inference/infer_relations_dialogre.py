from src.infering import EntityRelationInferer
from src.processing.dialogre_processing import DialogREDatasetTransformer
from src.paths import LOCAL_MODELS_PATH, LOCAL_PROCESSED_DATA_PATH, LOCAL_RAW_DATA_PATH

if __name__ == "__main__":
    
    dialogue_list = [
        "Speaker 1: A new place for a new Ross. I'm gonna have you and all the guys from work over once it's y'know, furnished.",
        "Speaker 2: I must say it's nice to see you back on your feet.",
        "Speaker 1: Well I am that. And that whole rage thing is definitely behind me.",
        "Speaker 2: I wonder if its time for you to rejoin our team at the museum?",
        "Speaker 1: Oh Donald that-that would be great. I am totally ready to come back to work. I—What? No! Wh… What are you doing?!!  GET OFF MY SISTER!!!!!!!!!!!!!"
        ]
    relation = {
        "y": "museum",
        "x": "Speaker 1",
        "rid": [
        28
        ],
        "r": [
        "per:place_of_work"
        ],
        "t": [
        "rejoin"
        ],
        "x_type": "PER",
        "y_type": "STRING"
    }
    
    ent_x, ent_y = relation['x'], relation['y']
    
    T2 = 0.32ee
    relation_type_count = 36
    bert_config_file = LOCAL_MODELS_PATH / "downloaded/bert-base/bert_config.json"
    vocab_file = LOCAL_MODELS_PATH / "downloaded/bert-base/vocab.txt"
    model_path=LOCAL_MODELS_PATH / "fine-tuned/bert-base-DialogRe/Unfrozen/24bs-1cls-2em5lr-20ep/model_best.pt"
    relation_label_dict = LOCAL_RAW_DATA_PATH / 'dialog-re/relation_label_dict.json'

    inferer = EntityRelationInferer(
        bert_config_file = bert_config_file, 
        vocab_file = vocab_file, 
        model_path = model_path, 
        relation_type_count = relation_type_count, 
        relation_label_dict = relation_label_dict,
        T2 = T2)
    
    rid_prediction, relation_label = inferer.infer_relations(' '.join(dialogue_list), ent_x, ent_y)
    
    print(rid_prediction, relation_label)