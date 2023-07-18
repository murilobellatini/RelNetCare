import json
import torch
import numpy as np


from src.config import device
from src.paths import LOCAL_RAW_DATA_PATH
from src.custom_dialogre.run_classifier import InputExample, convert_examples_to_features, getpred
from src.custom_dialogre.modeling import BertForSequenceClassificationWithExtraFeatures, BertConfig, BertForSequenceClassification
from src.custom_dialogre.tokenization import FullTokenizer

class EntityExtractor:
    """
    @todo: develop module with SpaCy
    """
    def extract(self):
        pass

class EntityRelationInferer:
    """
    A class for inferring relations between entities in a dialogue using a BERT model.
    """
    def __init__(self, bert_config_file, vocab_file, model_path, T2=0.4, relation_type_count=36, max_seq_length = 512, relation_label_dict=LOCAL_RAW_DATA_PATH / 'dialog-re-fixed-relations/relation_label_dict.json', do_lower_case=True, device=device):
        self.bert_config_file = bert_config_file
        self.vocab_file = vocab_file
        self.relation_label_dict = relation_label_dict
        self.model_path = model_path
        self.do_lower_case = do_lower_case
        self.max_seq_length = max_seq_length
        self.device = device
        self.relation_type_count = relation_type_count
        self.T2 = T2

        # Load model and tokenizer
        self.entity_extractor = EntityExtractor()
        self.bert_config = BertConfig.from_json_file(self.bert_config_file)
        self.model = BertForSequenceClassification(self.bert_config, 1, self.relation_type_count)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()  # Set model to evaluation mode
        self.model.to(self.device)  # Move model to device once
        self.tokenizer = FullTokenizer(vocab_file=self.vocab_file, do_lower_case=self.do_lower_case)
        self.label_dict = self._load_relation_label_dictionary()  # Load JSON file once

    def infer_relations(self, dialogue, src_entity, dst_entity):
        input_ids, segment_ids, input_mask = self._prepare_features(dialogue, src_entity, dst_entity)

        # Pass inputs through model
        with torch.no_grad():
            logits = self.model(input_ids, segment_ids, input_mask).detach().cpu().numpy()

        new_logits = self._normalize_logits(logits)

        # Get predictions from outputs
        predictions = getpred(
            result=new_logits,
            relation_type_count=self.relation_type_count,
            T1=0.5,
            T2=self.T2)
        
        # Shifts index to match 'rid'
        rid_prediction = predictions[0][0] + 1 
        
        relation_label = self._convert_rid_to_label(rid_prediction)

        return rid_prediction, relation_label
    
    def extract_entities(self, text):
        # You need to provide the logic here to extract entities from the text.
        # For now, this is a placeholder.
        return self.entity_extractor.extract(text)

    def _normalize_logits(self, logits):
        logits = np.asarray(logits)
        logits = list(1 / (1 + np.exp(-logits)))
        return logits

    def _prepare_features(self, dialogue, entity1, entity2):
        # Create an example using dialogue and entities
        example = InputExample(guid=None, text_a=dialogue, text_b=entity1, text_c=entity2, label=None)

        # Convert example to features and get tensors
        features = convert_examples_to_features([example], None, self.max_seq_length, self.tokenizer)[0]
        input_ids = torch.tensor([features[0].input_ids], dtype=torch.long).unsqueeze(0).to(self.device)
        segment_ids = torch.tensor([features[0].segment_ids], dtype=torch.long).unsqueeze(0).to(self.device)
        input_mask = torch.tensor([features[0].input_mask], dtype=torch.long).unsqueeze(0).to(self.device)

        return input_ids, segment_ids, input_mask

    def _load_relation_label_dictionary(self):
        with open(self.relation_label_dict, 'r') as file:
            data = json.load(file)

        # Convert keys back to integers
        label_dict = {int(k): v for k, v in data.items()}
        return label_dict
    
    def _convert_rid_to_label(self, rid:int):
        return self.label_dict[rid]




