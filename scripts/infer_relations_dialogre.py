from src.modelling import EntityRelationInferer
from src.paths import LOCAL_MODELS_PATH, LOCAL_PROCESSED_DATA_PATH

if __name__ == "__main__":
    relation_type_count = 36
    entity1, entity2 = "Charlie", "Joey"  # @todo: extract_entities(dialogue)
    
    dialogue = '\n'.join([
        "Speaker 1: It's so weird, how did Joey end up kissing Charlie last night? I thought you'd end up kissing Charlie.",
        "Speaker 2: Hey, I thought I'd end up kissing Charlie too ok? But SURPRISE!",
        "Speaker 3: I missed most of the party Charlie's a girl, right?",
        "Speaker 2: Yes, she is this new professor of my department that I did not kiss.",
        "Speaker 4: I don't know why Joey had to kiss her! I mean, of all the girls at the party, GOD!",
        "Speaker 2: Why do you care so much?",
        "Speaker 1: Yes Rachel, why do you care so much?",
        "Speaker 4:  Be-cause Ross is the father of my child! You know... and I... want him to hook up with lots of women! I just... All I'm saying is... I don't think that Joey and Charlie have anything in common.",
        "Speaker 2: Oh, I don't know, they seem to have a shared interest in each other's tonsils...",
        "Speaker 5: Wow, Joey and a professor! Can you imagine if they had kids and if the kids got her intelligence and Joey's raw sexual magnetism... Oh, those nerds will get laaaaaid!",
        "Speaker 4: All right, so... Ross, you're ok with all this? I mean...",
        "Speaker 2: Yeah, it's no big deal. I mean, I just met her and I'm fine with it...",
        "Speaker 2: Oh, God. I forgot how hot she was!"
    ])

    bert_config_file = LOCAL_MODELS_PATH / "downloaded/bert-base/bert_config.json"
    vocab_file = LOCAL_MODELS_PATH / "downloaded/bert-base/vocab.txt"
    model_path = LOCAL_MODELS_PATH / "fine-tuned/dialogre-fine-tuned/bert_base/model_best.pt"
    relation_label_dict = LOCAL_PROCESSED_DATA_PATH / 'dialog-re-fixed-relations/relation_label_dict.json'

    inferer = EntityRelationInferer(
        bert_config_file=bert_config_file, 
        vocab_file=vocab_file, 
        model_path=model_path, 
        relation_type_count=relation_type_count, 
        relation_label_dict=relation_label_dict)

    # Use the function
    predictions = inferer.infer_relations(dialogue, entity1, entity2)
    print(predictions)