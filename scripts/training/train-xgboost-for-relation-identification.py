from src.modelling import RelationModel


if __name__ == "__main__":
    relation_model = RelationModel(data_dir='dialog-re-binary-validated-enriched')
    relation_model.train_and_evaluate()