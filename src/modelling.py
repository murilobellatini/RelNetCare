import torch
from src.custom_dialogre.run_classifier import InputExample, convert_examples_to_features, getpred
from src.custom_dialogre.modeling import BertForSequenceClassification, BertConfig
from src.custom_dialogre.tokenization import FullTokenizer

class EntityRelationInferer:
    """
    A class for inferring relations between entities in a dialogue using a BERT model.

    This class leverages a BERT model to predict relationships between extracted entities within 
    a dialogue context. The primary components include the `extract_entities` method for identifying
    relevant entities in the text and the `infer_relations` method which utilizes the trained BERT model
    to infer the relationships between these entities.
    """
    def __init__(self, bert_config_file, vocab_file, model_path, do_lower_case=True, device='cpu'):
        self.bert_config_file = bert_config_file
        self.vocab_file = vocab_file
        self.model_path = model_path
        self.do_lower_case = do_lower_case
        self.device = device

        # Load model and tokenizer
        self.bert_config = BertConfig.from_json_file(self.bert_config_file)
        self.model = BertForSequenceClassification(self.bert_config, 1)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))  # Load from the saved model file
        self.tokenizer = FullTokenizer(vocab_file=self.vocab_file, do_lower_case=self.do_lower_case)

    def extract_entities(self, text):
        # You need to provide the logic here to extract entities from the text.
        # For now, this is a placeholder.
        entity1 = 'Entity1'
        entity2 = 'Entity2'
        return entity1, entity2

    def _prepare_features(self, dialogue, entity1, entity2):
        # Create an example using dialogue and entities
        example = InputExample(guid=None, text_a=dialogue, text_b=entity1, text_c=entity2, label=None)
        max_seq_length = 512  # adjust as per your model's configuration

        # Convert example to features and get tensors
        features = convert_examples_to_features([example], None, max_seq_length, self.tokenizer)[0]
        input_ids = torch.tensor([features[0].input_ids], dtype=torch.long).unsqueeze(0).to(self.device)
        segment_ids = torch.tensor([features[0].segment_ids], dtype=torch.long).unsqueeze(0).to(self.device)
        input_mask = torch.tensor([features[0].input_mask], dtype=torch.long).unsqueeze(0).to(self.device)

        return input_ids, segment_ids, input_mask

    def infer_relations(self, dialogue, entity1, entity2):
        input_ids, segment_ids, input_mask = self._prepare_features(dialogue, entity1, entity2)

        # Ensure model is in evaluation mode and move it to the correct device
        self.model.eval()
        self.model.to(self.device)

        # Pass inputs through model
        with torch.no_grad():
            outputs = self.model(input_ids, segment_ids, input_mask)

        # Get predictions from outputs
        predictions = getpred(outputs)[0][0]

        return predictions



