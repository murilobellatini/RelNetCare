import json
import spacy
import numpy as np
from transformers import pipeline
from fastcoref import spacy_component

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load the Spacy model
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe("fastcoref")

def resolve_coreferences(text):
    # Apply the pipeline to the text
    doc = nlp(      # for multiple texts use nlp.pipe
        text, 
        component_cfg={"fastcoref": {'resolve_text': True}}
        )
    # Return the text with resolved coreferences
    return doc._.resolved_text

def extract_features(text, entity1, entity2, dep_formats=None):
    if dep_formats is None:
        dep_formats = ['dep']

    # Parse the text using Spacy
    doc = nlp(text)

    # Prepare dictionary to store dependency paths in different formats
    dep_paths = {}

    # Extract the dependency paths for each requested format
    for dep_format in dep_formats:
        dep_paths[dep_format] = extract_dependency_paths(doc, entity1, entity2, dep_format)

    # Get the Spacy tokens for the two entities
    entity1_token = [token for token in doc if token.text == entity1][0]
    entity2_token = [token for token in doc if token.text == entity2][0]
    
    # Entity types
    entity1_type = entity1_token.ent_type_
    entity2_type = entity2_token.ent_type_

    # Part-of-speech tags
    entity1_pos = entity1_token.pos_
    entity2_pos = entity2_token.pos_

    # Surrounding words
    entity1_surrounding = [doc[i].text for i in range(max(entity1_token.i - 2, 0), min(entity1_token.i + 3, len(doc)))]
    entity2_surrounding = [doc[i].text for i in range(max(entity2_token.i - 2, 0), min(entity2_token.i + 3, len(doc)))]

    # TF-IDF scores
    tfidf = TfidfVectorizer()
    model = tfidf.fit_transform([text])
    scores = model.toarray()[0]
    entity1_tfidf = scores[tfidf.vocabulary_[entity1.lower()]]
    entity2_tfidf = scores[tfidf.vocabulary_[entity2.lower()]]

    # Semantic similarity
    semantic_similarity = cosine_similarity([entity1_token.vector], [entity2_token.vector])[0, 0]

    features = {
        'text': text,
        'entity1': entity1,
        'entity2': entity2,
        'dependency_paths': dep_paths,
        'entity1_type': entity1_type,
        'entity2_type': entity2_type,
        'entity1_pos': entity1_pos,
        'entity2_pos': entity2_pos,
        'entity1_surrounding': entity1_surrounding,
        'entity2_surrounding': entity2_surrounding,
        'entity1_tfidf': entity1_tfidf,
        'entity2_tfidf': entity2_tfidf,
        'semantic_similarity': semantic_similarity,
    }

    return features

def extract_dependency_paths(doc, entity1, entity2, dep_format='dep'):
    # Get the Spacy tokens for the two entities
    entity1_token = [token for token in doc if token.text == entity1][0]
    entity2_token = [token for token in doc if token.text == entity2][0]

    # Get the paths
    entity1_path = [token for token in entity1_token.ancestors]
    entity2_path = [token for token in entity2_token.ancestors]

    # Reverse the paths as Spacy provides ancestors in bottom-up order
    entity1_path = list(reversed(entity1_path))
    entity2_path = list(reversed(entity2_path))

    # Find the common ancestor (the latest common token in the paths)
    i = 0
    while i < len(entity1_path) and i < len(entity2_path) and entity1_path[i] == entity2_path[i]:
        i += 1

    # The dependency path will be the path from entity1 to the common ancestor 
    # and then the path from the common ancestor to entity2
    dependency_path = entity1_path[i-1:] + [entity1_token, entity2_token] + list(reversed(entity2_path[i-1:]))

    if dep_format == 'dep':
        return [token.dep_ for token in dependency_path]
    elif dep_format == 'pos':
        return [token.pos_ for token in dependency_path]
    else:  # raw
        return [token.text for token in dependency_path]

def process_dialogue(dialogue, entities):
    # Concatenate all dialogue turns into a single text
    text = '\n'.join([f"{turn['role']}: {turn['content']}" for turn in dialogue])
    resolved_text = resolve_coreferences(text)
    resolved_dialogue = resolved_text.split('\n')

    # Extract features for each pair of entities
    output_dicts = []
    for entity1, entity2 in entities:
        for d in resolved_dialogue:
            if entity1 in d and entity2 in d:
                features = extract_features(d, entity1, entity2, dep_formats=['dep', 'pos', 'tokens'])
                output_dicts.append(features)
                
    return output_dicts


def default(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    raise TypeError


if __name__ == '__main__':
    dialogue = [
        {"role": "User", "content": "Alice moved to Munich."},
        {"role": "Agent", "content": "That's interesting. What does she do there?"},
        {"role": "User", "content": "She works for Google."},
    ]

    entities = [
        ("Alice", "Munich"),
        ("Alice", "Google"),
    ]

    output_dicts = process_dialogue(dialogue, entities)

    # Now you can use this in your json.dumps method:
    print(json.dumps(output_dicts, indent=2, default=default))

