import torch
from src.paths import LOCAL_MODELS_PATH
from src.custom_dialogre.run_classifier import InputExample, convert_examples_to_features, getpred
from src.custom_dialogre.modeling import BertForSequenceClassification, BertConfig
from src.custom_dialogre.tokenization import FullTokenizer

def extract_entities(text):
    # You need to provide the logic here to extract entities from the text.
    # For now, this is a placeholder.
    entity1 = 'Entity1'
    entity2 = 'Entity2'
    return entity1, entity2

def infer_relations(dialogue, entity1, entity2, model, tokenizer, device='cpu'):
    max_seq_length = 512  # adjust as per your model's configuration
    
    # Create an example using dialogue and entities
    example = InputExample(guid=None, text_a=dialogue, text_b=entity1, text_c=entity2, label=None)
    
    # Convert example to features
    features = convert_examples_to_features([example], None, max_seq_length, tokenizer)
    
    # Get the tensors from the features
    input_ids = torch.tensor([f.input_ids for f in features[0]], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features[0]], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features[0]], dtype=torch.long)
    
    # Add a batch dimension and move tensors to the correct device
    input_ids = input_ids.unsqueeze(0).to(device)
    segment_ids = segment_ids.unsqueeze(0).to(device)
    input_mask = input_mask.unsqueeze(0).to(device)

    # Ensure model is in evaluation mode and move it to the correct device
    model.eval()
    model.to(device)

    # Pass inputs through model
    with torch.no_grad():
        outputs = model(input_ids, segment_ids, input_mask)

    # Get predictions from outputs
    predictions = getpred(outputs)

    # Convert predictions to labels
    # We would typically convert numeric predictions to their corresponding labels here, using the label mapping.

    return predictions

if __name__ == "__main__":
    dialogue = [
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
        ]   
    entity1, entity2 = "professor", "Charlie" # @todo: extract_entities(dialogue)

    bert_config_file = LOCAL_MODELS_PATH / "downloaded/bert-tiny/bert_config.json"
    vocab_file = LOCAL_MODELS_PATH /"downloaded/bert-tiny/vocab.txt"
    model_path = LOCAL_MODELS_PATH / "fine-tuned/dialogre-fine-tuned/bert_tiny/model_best.pt"
    
    do_lower_case = True
    bert_config = BertConfig.from_json_file(bert_config_file)
    model = BertForSequenceClassification(bert_config, 1)
    model.load_state_dict(torch.load(model_path))  # Load from the saved model file
    tokenizer = FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)
        
    # Use the function
    predictions = infer_relations(dialogue, entity1, entity2, model, tokenizer)
    print(predictions)