import json
from src.processing.text_preprocessing import CoreferenceResolver, FeatureExtractor


if __name__ == '__main__':
    dialogue = [
        "User: Alice moved to Munich.",
        "Agent: That's interesting. What does she do there?",
        "User: She works for Google.",
    ]

    entity_pairs = [
        ("Alice", "Munich"),
        ("Alice", "Google"),
    ]

    coref_resolver = CoreferenceResolver()
    extractor = FeatureExtractor()

    resolved_dialogue = coref_resolver.process_dialogue(dialogue)
    output_dicts = extractor.process_dialogue(resolved_dialogue, entity_pairs)
    print(json.dumps(output_dicts, indent=2, default=float))
