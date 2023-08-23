import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LLMTransformationConfig:
    def __init__(self,
                 max_turns=None,
                 max_speakers=None,
                 skip_empty_triples=False,
                 balance_empty_dialogues=True,
                 parse_subdialogues=False):
        
        self.all_relations = {
            "positive_impression", "negative_impression", "acquaintance", 
            "alumni", "boss", "subordinate", "client", "dates", "friends", 
            "girl/boyfriend", "neighbor", "roommate", "children", "other_family", 
            "parents", "siblings", "spouse", "place_of_residence", "visited_place", 
            "origin", "employee_or_member_of", "schools_attended", "works", "age", 
            "date_of_birth", "major", "place_of_work", "title", "alternate_names", 
            "pet", "residents_of_place", "visitors_of_place", "employees_or_members", 
            "students", "unanswerable"
        }
        self.allowed_relations = {
            "acquaintance", "children", "other_family", "parents", 
            "siblings", "spouse", "place_of_residence", "visited_place", 
            "pet", "residents_of_place", "visitors_of_place"
            }
        
        self.skip_relations = self.filter_skip_relations()
        self.total_relation_count = len(self.skip_relations)

        self.ds_type = ""
        self.ds_root = f"dialog-re-llama{self.ds_type}"
        self.input_dir = "/home/murilo/RelNetCare/data/raw/dialog-re"
        self.file_sets = [['train', 'dev'], ['test']]
        self.max_turns = max_turns
        self.max_speakers = max_speakers
        self.skip_empty_triples = skip_empty_triples
        self.balance_empty_dialogues = balance_empty_dialogues
        self.parse_subdialogues = parse_subdialogues
        self.output_dir = self.get_output_folder_name()
        self.preprompt = self.generate_preprompt()
        
        print("output_dir=",self.output_dir)

    def filter_skip_relations(self):
        return [r for r in self.all_relations if r not in self.allowed_relations]

    def get_output_folder_name(self):
        parts = [self.ds_root, f"{len(self.allowed_relations)}cls"]
        if self.max_turns:
            parts.append(f"{self.max_turns}trn")
        if self.max_speakers:
            parts.append(f"{self.max_speakers}spkr")
        if self.skip_empty_triples:
            parts.append(f"skipEmptyPairs")
        if self.balance_empty_dialogues:
            parts.append(f"balPairs")
        if self.parse_subdialogues:   # assuming you have this attribute in your config
            parts.append(f"parseSubDlgs")

        return os.path.join("/home/murilo/RelNetCare/data/processed", "-".join(parts))

    def generate_preprompt(self, add_one_shot=False):
        
        one_shot = """
Input:
[
"User: My daughter, Emma, recently moved to London.",
"Agent: That's exciting! Does she like it there?",
"User: Yes, she loves it! She even adopted a cat named Whiskers.",
]

Output:
[
{{"x": "User", "x_type": "PERSON", "y": "Emma", "y_type": "PERSON", "r": "children"}},
{{"x": "Emma", "x_type": "PERSON", "y": "London", "y_type": "GPE", "r": "place_of_residence"}},
{{"x": "London", "x_type": "GPE", "y": "Emma", "y_type": "PERSON", "r": "residents_of_place"}},
{{"x": "Emma", "x_type": "PERSON", "y": "Whiskers", "y_type": "ANIMAL", "r": "pet"}},
{{"x": "Whiskers", "x_type": "ANIMAL", "y": "Emma", "y_type": "PERSON", "r": "pet"}},
]
"""

        preprompt = f"""
Extract personal relevant entities, and their relations. Return only the jsonl format list .

Ontology: 
- relations: {str(self.allowed_relations).replace("'", '"')}
- types: {{"ORG", "GPE", "PERSON", "DATE", "EVENT", “ANIMAL”}}
{one_shot if add_one_shot else ""}
Input:
"""
        return preprompt