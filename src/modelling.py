import json
import torch
from torchsummary import summary

from src.config import device
from src.paths import LOCAL_RAW_DATA_PATH
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
    def __init__(self, bert_config_file, vocab_file, model_path, relation_type_count=36, max_seq_length = 512, relation_label_dict=LOCAL_RAW_DATA_PATH / 'dialog-re-fixed-relations/relation_label_dict.json', do_lower_case=True, device=device):
        self.bert_config_file = bert_config_file
        self.vocab_file = vocab_file
        self.relation_label_dict = relation_label_dict
        self.model_path = model_path
        self.do_lower_case = do_lower_case
        self.max_seq_length = max_seq_length
        self.device = device
        self.relation_type_count = relation_type_count

        # Load model and tokenizer
        self.bert_config = BertConfig.from_json_file(self.bert_config_file)
        self.model = BertForSequenceClassification(self.bert_config, 1, self.relation_type_count)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))  # Load from the saved model file
        self.tokenizer = FullTokenizer(vocab_file=self.vocab_file, do_lower_case=self.do_lower_case)

    def infer_relations(self, dialogue, entity1, entity2):
        input_ids, segment_ids, input_mask = self._prepare_features(dialogue, entity1, entity2)

        # Ensure model is in evaluation mode and move it to the correct device
        self.model.eval()
        self.model.to(self.device)

        # Pass inputs through model
        with torch.no_grad():
            outputs = self.model(input_ids, segment_ids, input_mask)

        # Get predictions from outputs
        predictions = getpred(outputs, self.relation_type_count)[0][0]
        
        labels = self._rid_to_label(predictions)

        return predictions, labels
    
    def extract_entities(self, text):
        # You need to provide the logic here to extract entities from the text.
        # For now, this is a placeholder.
        entity1 = 'Entity1'
        entity2 = 'Entity2'
        return entity1, entity2

    def _prepare_features(self, dialogue, entity1, entity2):
        # Create an example using dialogue and entities
        example = InputExample(guid=None, text_a=dialogue, text_b=entity1, text_c=entity2, label=None)

        # Convert example to features and get tensors
        features = convert_examples_to_features([example], None, self.max_seq_length, self.tokenizer)[0]
        input_ids = torch.tensor([features[0].input_ids], dtype=torch.long).unsqueeze(0).to(self.device)
        segment_ids = torch.tensor([features[0].segment_ids], dtype=torch.long).unsqueeze(0).to(self.device)
        input_mask = torch.tensor([features[0].input_mask], dtype=torch.long).unsqueeze(0).to(self.device)

        return input_ids, segment_ids, input_mask

    def _load_label_dict(self):
        with open(self.relation_label_dict, 'r') as file:
            data = json.load(file)

        # Convert keys back to integers
        label_dict = {int(k): v for k, v in data.items()}
        return label_dict
    
    def _rid_to_label(self, rid:int):
        label_dict = self._load_label_dict()
        return label_dict[rid]




