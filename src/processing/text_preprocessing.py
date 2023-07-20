import nltk
import spacy
from tqdm import tqdm
from fastcoref import spacy_component
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


from src.processing.distance_computation import TurnDistanceCalculator, DistanceComputer

            
class DialogueEnricher:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        nltk.download('punkt')
        self.txt_pos_tracker = TextPositionTracker(self.nlp)
        self.entity_list_flattener = EntityListFlattener()
        self.feature_extractor = SpacyFeatureExtractor(self.nlp)
        self.turn_distance_calculator = TurnDistanceCalculator()
        self.distance_computer = DistanceComputer(self.txt_pos_tracker, self.entity_list_flattener, self.feature_extractor, self.turn_distance_calculator)

    def enrich(self, dialogues_relations):
        processed_dialogues = self._compute_dialogues_relations_distances(dialogues_relations)
        self._validate_processed_dialogues(dialogues_relations, processed_dialogues)
        processed_dialogues = self._filter_processed_dialogues(processed_dialogues)
        return processed_dialogues

    def _compute_dialogues_relations_distances(self, dialogues_relations):
        processed_dialogues = []
        for dialogue, relations in tqdm(dialogues_relations):
            relations_with_distances = self.distance_computer.compute_distances(dialogue, relations)
            processed_dialogues.append((dialogue, relations_with_distances))
        return processed_dialogues

    def _validate_processed_dialogues(self, dialogues_relations, processed_dialogues):
        # Assert that 'data' and 'new_data' have the same length
        assert len(dialogues_relations) == len(processed_dialogues), "Data and new data have different lengths"

        # Assert that 'data' and 'new_data' have the same relation count for each dialogue
        for original_dialogue, new_dialogue in zip(dialogues_relations, processed_dialogues):
            assert len(original_dialogue[1]) == len(new_dialogue[1]), "Original and new dialogues have different relation counts"

    def _filter_processed_dialogues(self, processed_dialogues):
        # Instead of filtering out relations that do not contain 'min_words_distance', set it to None
        for i, (dialogue, relations) in enumerate(processed_dialogues):
            for r in relations:
                if 'min_words_distance' not in r:
                    r['min_words_distance'] = -1
        return processed_dialogues



class TextPositionTracker:
    def __init__(self, nlp):
        self.nlp = nlp

    def tokenize_text(self, text: str, terms: List[str]) -> Tuple[spacy.tokens.Doc, Dict[str, List[Tuple[int, int]]], Dict[str, List[Tuple[int, int]]]]:
        doc = self.nlp(text)
        tokens = [token.text.lower() for token in doc]  # Convert tokens to lowercase
        lemma_tokens = [token.lemma_.lower() for token in doc]  # Convert lemmatized tokens to lowercase
        token_positions = {term: [] for term in terms}
        char_positions = {term: [] for term in terms}

        for term in terms:
            term_tokens = term.lower().split()  # Convert term to lowercase
            term_len = len(term_tokens)

            for i in range(len(tokens) - term_len + 1):
                if tokens[i:i+term_len] == term_tokens:
                    token_positions[term].append((i, i+term_len))
                    char_positions[term].append((doc[i].idx, doc[i+term_len-1].idx + len(doc[i+term_len-1])))

            # If no match found, then use lemmatized version
            if not token_positions[term]:
                term_tokens = [token.lemma_ for token in self.nlp(term.lower())]
                term_len = len(term_tokens)

                for i in range(len(lemma_tokens) - term_len + 1):
                    if lemma_tokens[i:i+term_len] == term_tokens:
                        token_positions[term].append((i, i+term_len))
                        char_positions[term].append((doc[i].idx, doc[i+term_len-1].idx + len(doc[i+term_len-1])))
            
            # If still no match found, try fuzzy matching
            if not token_positions[term]:
                term_tokens = term.lower().split()
                term_len = len(term_tokens)

                for i in range(len(tokens) - term_len + 1):
                    if fuzz.partial_ratio(' '.join(tokens[i:i+term_len]), term) >= self.fuzzy_threshold:
                        token_positions[term].append((i, i+term_len))
                        char_positions[term].append((doc[i].idx, doc[i+term_len-1].idx + len(doc[i+term_len-1])))

        return doc, token_positions, char_positions

class EntityListFlattener:
    @staticmethod
    def get_entities(relations: List[dict]) -> List[str]:
        entities = set()
        for relation in relations:
            entities.add(relation['x'])
            entities.add(relation['y'])
        return list(entities)

 
class SpacyFeatureExtractor:
    def __init__(self, nlp):
        self.nlp = nlp

    def get_spacy_features(self, doc: spacy.tokens.Doc, x_span: Tuple[int, int], y_span: Tuple[int, int]) -> Dict[str, str]:
        x_token = doc[x_span[0]]
        y_token = doc[y_span[1] - 1]
        
        return {
            "x_pos": x_token.pos_,
            "x_dep": x_token.dep_,
            "x_tag": x_token.tag_,
            "y_pos": y_token.pos_,
            "y_dep": y_token.dep_,
            "y_tag": y_token.tag_,
        }
        

class CoreferenceResolver:
    """
    Handles the resolution of coreferences in a given text."""

    def __init__(self, model='en_core_web_sm'):
        self.nlp = spacy.load(model)
        self.nlp.add_pipe("fastcoref")

    def resolve_coreferences(self, text):
        doc = self.nlp(text, component_cfg={"fastcoref": {'resolve_text': True}})
        return doc._.resolved_text

    def process_dialogue(self, dialogue):
        text = '\n'.join(dialogue)
        resolved_text = self.resolve_coreferences(text)
        return resolved_text.split('\n')


class DependencyPathFinder:
    """Finds the dependency path between two entities in a given document."""
    
    def __init__(self, model='en_core_web_sm'):
        self.nlp = spacy.load(model)

    def find_dependency_path(self, doc, entity1, entity2, dep_format='dep'):
        entity1_token = [token for token in doc if token.text == entity1][0]
        entity2_token = [token for token in doc if token.text == entity2][0]

        entity1_path = list(reversed([token for token in entity1_token.ancestors]))
        entity2_path = list(reversed([token for token in entity2_token.ancestors]))

        i = 0
        while i < len(entity1_path) and i < len(entity2_path) and entity1_path[i] == entity2_path[i]:
            i += 1

        dependency_path = entity1_path[i-1:] + [entity1_token, entity2_token] + list(reversed(entity2_path[i-1:]))

        if dep_format == 'dep':
            return [token.dep_ for token in dependency_path]
        elif dep_format == 'pos':
            return [token.pos_ for token in dependency_path]
        else:  # raw
            return [token.text for token in dependency_path]


class FeatureExtractor:
    """Extracts the features from a given text for two entities."""

    def __init__(self, model='en_core_web_sm', dep_formats=None):
        self.nlp = spacy.load(model)
        self.dep_formats = dep_formats or ['dep']
        self.tfidf = TfidfVectorizer()
        self.dependencyPathFinder = DependencyPathFinder(model)

    def extract_features(self, text, entity1, entity2):
        doc = self.nlp(text)

        dep_paths = {
            dep_format: self.dependencyPathFinder.find_dependency_path(doc, entity1, entity2, dep_format)
            for dep_format in self.dep_formats
        }

        entity1_token = [token for token in doc if token.text == entity1][0]
        entity2_token = [token for token in doc if token.text == entity2][0]

        model = self.tfidf.fit_transform([text])
        scores = model.toarray()[0]
        entity1_tfidf = scores[self.tfidf.vocabulary_[entity1.lower()]]
        entity2_tfidf = scores[self.tfidf.vocabulary_[entity2.lower()]]

        semantic_similarity = cosine_similarity([entity1_token.vector], [entity2_token.vector])[0, 0]

        return {
            'text': text,
            'entity1': entity1,
            'entity2': entity2,
            'dependency_paths': dep_paths,
            'entity1_type': entity1_token.ent_type_,
            'entity2_type': entity2_token.ent_type_,
            'entity1_pos': entity1_token.pos_,
            'entity2_pos': entity2_token.pos_,
            'entity1_surrounding': [doc[i].text for i in range(max(entity1_token.i - 2, 0), min(entity1_token.i + 3, len(doc)))],
            'entity2_surrounding': [doc[i].text for i in range(max(entity2_token.i - 2, 0), min(entity2_token.i + 3, len(doc)))],
            'entity1_tfidf': entity1_tfidf,
            'entity2_tfidf': entity2_tfidf,
            'semantic_similarity': semantic_similarity,
        }

    def process_dialogue(self, resolved_dialogue, entities):
        output_dicts = []
        for entity1, entity2 in entities:
            for d in resolved_dialogue:
                if entity1 in d and entity2 in d:
                    features = self.extract_features(d, entity1, entity2)
                    output_dicts.append(features)
        return output_dicts

