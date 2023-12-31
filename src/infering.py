import re
import json
import spacy
import torch
import itertools
import numpy as np
import xgboost as xgb
from typing import List, Tuple, Dict
from pathlib import Path

from src.config import device
from src.paths import LOCAL_RAW_DATA_PATH, LOCAL_MODELS_PATH, LOCAL_PROCESSED_DATA_PATH
from src.custom_dialogre.run_classifier import InputExample, convert_examples_to_features, getpred
from src.custom_dialogre.modeling import BertForSequenceClassificationWithExtraFeatures, BertConfig, BertForSequenceClassification
from src.custom_dialogre.tokenization import FullTokenizer
from src.processing.text_preprocessing import DialogueEnricher, CoreferenceResolver
from src.modelling import InferenceRelationModel
from src.processing.neo4j_operations import DialogueGraphPersister

class EntityExtractor:
    def __init__(self, spacy_model='en_core_web_sm', extract_dialogue_speakers=True):
        self.nlp = spacy.load(spacy_model)
        self.extract_dialogue_speakers = extract_dialogue_speakers
    
    def process(self, text, ignore_types=[]):
        entities = self._extract_entities(text, ignore_types=ignore_types)
        entity_pairs = self._get_entity_permutations(entities)
        enriched_entities = self._enrich_entities(entity_pairs)
        return enriched_entities
    
    def _extract_entities(self, text, ignore_types=[]):
        """
        Extract entities from text using Spacy.
        """
        doc = self.nlp(text)
        entities = set([(ent.text, ent.label_) for ent in doc.ents if ent.label_ not in ignore_types])
        if self.extract_dialogue_speakers:
            pattern = r'^(.*?):'
            speakers = re.findall(pattern, text, re.MULTILINE)
            entities.update((speaker, 'PERSON') for speaker in speakers)
        return entities

    def _get_entity_permutations(self, entities):
        """
        Compute all permutations of entities.
        """
        entity_permutations = list(itertools.permutations(entities, 2))
        return entity_permutations

    def _enrich_entities(self, entity_pairs):
        enriched_entities = []
        for x, y in entity_pairs:
            enriched_entity = {
                'x': x[0],
                'x_type': x[1],
                'y': y[0],
                'y_type': y[1]
            }
            enriched_entities.append(enriched_entity)
        return enriched_entities

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
        self.model = BertForSequenceClassification(self.bert_config, 1, 36)
        self.model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

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
            relation_type_count=36,
            T1=0.5,
            T2=self.T2)
        
        # Shifts index to match 'rid'
        try:
            rid_prediction = predictions[0][0] + 1 
        except Exception as e:
            print(f"Error {e}...\nDefaulting rid to 37 (unanswerable/no_relation)")
            rid_prediction = 37
        
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
        return self.label_dict.get(rid, 'no_relation')


class DialogRelationInferer(EntityRelationInferer):
    def __init__(self, bert_config_file, vocab_file, model_path, relation_type_count, relation_label_dict, T2):
        super().__init__(
            bert_config_file=bert_config_file,
            vocab_file=vocab_file, 
            model_path=model_path, 
            T2=T2, 
            relation_type_count=relation_type_count, 
            relation_label_dict=relation_label_dict 
            )

    def perform_inference(self, enriched_dialogues, pred_labels):
        dialogue_list, relations = enriched_dialogues
        assert len(relations) == len(pred_labels)
        for i, r in enumerate(relations):
            r['r_bool'] = pred_labels[i]
            r['t'] = "[]"
            if pred_labels[i] != 1:
                continue
            ent_x, ent_y = r['x'], r['y']
            rid_prediction, relation_label = self.infer_relations(' '.join(dialogue_list), ent_x, ent_y)
            r['rid'] = [rid_prediction]
            r['r'] = [relation_label] # @TODO: assess if relation type must be removed from label (before `:`)

        return (dialogue_list, relations)


class CustomTripletExtractor:
    """
    A class for inferring triplets using a custom pipeline.

        Steps:
        1. Entity Extraction: Utilizes SpaCy for entity extraction.
        2. Feature Extraction: Computes word distance and other relevant features.
        3. Relation Identification: Employs a custom XGBoost model.
        4. Relation Classification: Utilizes a BERT classifier trained on DialogRE.
    """
    def __init__(self,
                    bert_config_file=LOCAL_MODELS_PATH / "downloaded/bert-base/bert_config.json",
                    vocab_file=LOCAL_MODELS_PATH / "downloaded/bert-base/vocab.txt",
                    model_path=LOCAL_MODELS_PATH / "fine-tuned/bert-base-DialogRe{RELATION_TYPE_COUNT}/Unfrozen/24bs-1cls-3em5lr-20ep/model_best.pt",
                    relation_type_count=11,
                    relation_label_dict=LOCAL_RAW_DATA_PATH / 'dialog-re/relation_label_dict.json',
                    T2=0.32,
                    relation_identification_thresh=0.75,
                    apply_coref_resolution=False, # turned off since not present in train set of `InferenceRelationModel`
                    ner_model='en_core_web_trf'
                    ):
        # print("Initializing CustomTripletExtractor...")
        model_path_str = str(model_path)
        
        if relation_type_count == 36:
            formatted_model_path = model_path_str.format(RELATION_TYPE_COUNT='')
        else:
            formatted_model_path = model_path_str.format(RELATION_TYPE_COUNT=f'{relation_type_count}cls')
            relation_label_dict = LOCAL_PROCESSED_DATA_PATH / f'dialog-re-{relation_type_count}cls/relation_label_dict.json'
            
            
            
        model_path = Path(formatted_model_path) 

        self.skip_rel_ident = 'WithNoRelation' in str(model_path)
        self.apply_coref_resolution = apply_coref_resolution
        if apply_coref_resolution:
            self.coref_resolver = CoreferenceResolver()
        self.entity_extractor = EntityExtractor(spacy_model=ner_model)
        self.feature_enricher = DialogueEnricher()
        self.relation_identifier = InferenceRelationModel(data_dir='dialog-re-binary-validated-enriched', threshold=relation_identification_thresh)
        self.relation_classifier = DialogRelationInferer(
            bert_config_file=bert_config_file,
            vocab_file=vocab_file,
            model_path=model_path,
            relation_type_count=relation_type_count,
            relation_label_dict=relation_label_dict,
            T2=T2
        )
        # self.processor = DialogueGraphPersister('pipeline')
        # print("CustomTripletExtractor init successfully concluded!")

    def extract_triplets(self, dialogue) -> List[Dict]:
        if self.apply_coref_resolution:
            dialogue = self.coref_resolver.process_dialogue(dialogue)
            
        # 1. extract entities
        entity_pairs = self.entity_extractor.process('\n'.join(dialogue), ignore_types=['CARDINAL'])
        dialogues = [(dialogue, entity_pairs)]
        enriched_dialogues = self.feature_enricher.enrich(dialogues)
        # 2. identify relations
        if self.skip_rel_ident:
            # relation identification is done during classifications step
            pred_labels = np.ones((len(entity_pairs),)) 
        else:
            pred_labels = self.relation_identifier.get_predicted_labels(enriched_dialogues)
            
        pred_labels = [r for r in pred_labels if not r.get('relation') in ['not_found', 'no_relation', 'null_relation']]

        # 3. classify relations
        dialogue, predicted_relations = self.relation_classifier.perform_inference(enriched_dialogues[0], pred_labels)
        return predicted_relations
    
    def dump_to_neo4j(self, dialogue, predicted_relations) -> None:
        print("Dumping triplets to Neo4j...")
        self.processor.process_dialogue(dialogue, predicted_relations)
        self.processor.close_connection()
        print("Triplets dumped to Neo4j successfully.")
