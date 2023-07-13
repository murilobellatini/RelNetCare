from src.modelling import EntityRelationInferer
from src.paths import LOCAL_MODELS_PATH, LOCAL_PROCESSED_DATA_PATH, LOCAL_RAW_DATA_PATH

if __name__ == "__main__":
    relation_type_count = 37
    entity1, entity2 = "Chandler Bing", "Tom Gordon"  # @todo: extract_entities(dialogue)
    
    dialogue = [
   "Speaker 1: It's been an hour and not one of my classmates has shown up! I tell you, when I actually die some people are gonna get seriously haunted!",
   "Speaker 2: There you go! Someone came!",
   "Speaker 1: Ok, ok! I'm gonna go hide! Oh, this is so exciting, my first mourner!",
   "Speaker 3: Hi, glad you could come.",
   "Speaker 2: Please, come in.",
   "Speaker 4: Hi, you're Chandler Bing, right? I'm Tom Gordon, I was in your class.",
   "Speaker 2: Oh yes, yes... let me... take your coat.",
   "Speaker 4: Thanks... uh... I'm so sorry about Ross, it's...",
   "Speaker 2: At least he died doing what he loved... watching blimps.",
   "Speaker 1: Who is he?",
   "Speaker 2: Some guy, Tom Gordon.",
   "Speaker 1: I don't remember him, but then again I touched so many lives.",
   "Speaker 3: So, did you know Ross well?",
   "Speaker 4: Oh, actually I barely knew him. Yeah, I came because I heard Chandler's news. D'you know if he's seeing anyone?",
   "Speaker 3: Yes, he is. Me.",
   "Speaker 4: What? You... You... Oh! Can I ask you a personal question? Ho-how do you shave your beard so close?",
   "Speaker 2: Ok Tommy, that's enough mourning for you! Here we go, bye bye!!",
   "Speaker 4: Hey, listen. Call me.",
   "Speaker 2: Ok!"
  ]
    dialogue = ' \n'.join(dialogue)
    
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

    # Use the function
    predictions = inferer.infer_relations(dialogue, entity1, entity2)
    print(predictions)