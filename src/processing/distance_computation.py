from typing import List


class DistanceComputer:
    def __init__(self, txt_pos_tracker, entity_extractor, feature_extractor, turn_distance_calculator):
        self.txt_pos_tracker = txt_pos_tracker
        self.entity_extractor = entity_extractor
        self.feature_extractor = feature_extractor
        self.turn_distance_calculator = turn_distance_calculator

    def compute_distances(self, dialogue: List[str], relations: List[dict]) -> List[dict]:
        dialogue_str = ' '.join(dialogue)
        entities = self.entity_extractor.get_entities(relations)
        doc, entity_token_positions, entity_char_positions, matched_entities = self.txt_pos_tracker.find_term_positions(dialogue_str, entities)

        for relation in relations:
            x = relation['x']
            y = relation['y']

            x_match = matched_entities[x]
            y_match = matched_entities[y]

            relation['x'] = x_match
            relation['y'] = y_match

            x_token_positions = entity_token_positions[x]
            y_token_positions = entity_token_positions[y]
            x_char_positions = entity_char_positions[x]
            y_char_positions = entity_char_positions[y]
            
            # Additional check to avoid KeyError
            if not (x_token_positions and y_token_positions):
                continue

            min_distance = min([abs(x[1] - y[0]) for x in x_token_positions for y in y_token_positions])

            x_min_token = min([x[0] for x in x_token_positions])
            y_min_token = min([y[0] for y in y_token_positions])
            x_max_token = max([x[1] for x in x_token_positions])
            y_max_token = max([y[1] for y in y_token_positions])

            if x_min_token < y_min_token:
                relation["x_token_span"] = (x_min_token, x_max_token)
                relation["y_token_span"] = (y_min_token, y_max_token)
            else:
                relation["x_token_span"] = (y_min_token, y_max_token)
                relation["y_token_span"] = (x_min_token, x_max_token)

            x_min_char = min([x[0] for x in x_char_positions])
            y_min_char = min([y[0] for y in y_char_positions])
            x_max_char = max([x[1] for x in x_char_positions])
            y_max_char = max([y[1] for y in y_char_positions])

            if x_min_char < y_min_char:
                relation["x_char_span"] = (x_min_char, x_max_char)
                relation["y_char_span"] = (y_min_char, y_max_char)
            else:
                relation["x_char_span"] = (y_min_char, y_max_char)
                relation["y_char_span"] = (x_min_char, x_max_char)

            relation["min_words_distance"] = min_distance
            relation["min_words_distance_pct"] = min_distance / len(dialogue_str)
            
            relation['spacy_features'] = self.feature_extractor.get_spacy_features(doc, relation["x_token_span"], relation["y_token_span"])

        relations = self.turn_distance_calculator.compute_turn_distance(dialogue, relations)

        return relations


class TurnDistanceCalculator:
    @staticmethod
    def compute_turn_distance(dialogue: List[str], relations: List[dict]) -> List[dict]:
        for relation in relations:
            x = relation['x']
            y = relation['y']

            x_turn = [i for i, turn in enumerate(dialogue) if x in turn]

            y_turn = [i for i, turn in enumerate(dialogue) if y in turn]

            if x_turn and y_turn:
                relation["min_turn_distance"] = min([abs(i - j) for i in x_turn for j in y_turn])
                relation["min_turn_distance_pct"] = relation["min_turn_distance"] / len('\n'.join(dialogue))
            
        return relations
