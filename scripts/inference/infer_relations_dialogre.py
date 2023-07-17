from src.infering import EntityRelationInferer
from src.processing.dialogre_processing import DialogREDatasetTransformer
from src.paths import LOCAL_MODELS_PATH, LOCAL_PROCESSED_DATA_PATH, LOCAL_RAW_DATA_PATH

if __name__ == "__main__":
    
    dt = DialogREDatasetTransformer(LOCAL_RAW_DATA_PATH / 'dialog-re')
    df = dt.load_data_to_dataframe().explode('Relations')
    tmp = df[df.Origin == 'test']

    bert_config_file = LOCAL_MODELS_PATH / "downloaded/bert-base/bert_config.json"
    vocab_file = LOCAL_MODELS_PATH / "downloaded/bert-base/vocab.txt"
    model_path=LOCAL_MODELS_PATH / "fine-tuned/bert-base-dialog-re/Unfrozen/24bs-1cls-3em5lr-20ep/model_best.pt"
    relation_label_dict = LOCAL_RAW_DATA_PATH / 'dialog-re/relation_label_dict.json'
    relation_type_count = 36
    T2 = 0.32

    inferer = EntityRelationInferer(
        bert_config_file=bert_config_file, 
        vocab_file=vocab_file, 
        model_path=model_path, 
        relation_type_count=relation_type_count, 
        relation_label_dict=relation_label_dict,
        T2=T2)
    tmp['pred'] = tmp.apply(lambda row: inferer.infer_relations(' '.join(row['Dialogue']), row['Relations']['x'], row['Relations']['y']), axis=1)

    
    print('finished')