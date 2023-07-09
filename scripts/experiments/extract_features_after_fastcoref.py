import json
import spacy
import numpy as np
from fastcoref import spacy_component
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


if __name__ == '__main__':
    dialogue = [
        "User: Alice moved to Munich.",
        "Agent: That's interesting. What does she do there?",
        "User: She works for Google.",
    ]

    entities = [
        ("Alice", "Munich"),
        ("Alice", "Google"),
    ]

    coref_resolver = CoreferenceResolver()
    extractor = FeatureExtractor()

    resolved_dialogue = coref_resolver.process_dialogue(dialogue)
    output_dicts = extractor.process_dialogue(resolved_dialogue, entities)
    print(json.dumps(output_dicts, indent=2, default=float))
