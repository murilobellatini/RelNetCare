import re
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LLMTransformationConfig:
    def __init__(self,
                 max_turns=None,
                 max_speakers=None,
                 cls_task_only=False,
                 balance_empty_dialogues=False,
                 rebalance_empty_dialogues=False,
                 replace_skipped_with_others=False,
                 skip_empty_triples=False,
                 parse_subdialogues=False,
                 rewrite_keys=True,
                 add_one_shot=False,
                 instruction_type="A",
                 ignore_relation_filter=False):
        
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
        
        self.ignore_relation_filter = ignore_relation_filter
        
        if self.ignore_relation_filter:
            self.allowed_relations = self.all_relations
        
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
        self.rebalance_empty_dialogues = rebalance_empty_dialogues
        self.parse_subdialogues = parse_subdialogues
        self.replace_skipped_with_others = replace_skipped_with_others
        self.rewrite_keys = False if cls_task_only else rewrite_keys
        self.add_one_shot = False if cls_task_only else add_one_shot
        self.instruction_type = 'clsTskOnl' if cls_task_only else instruction_type
        self.replacement_dict = {
        'x': 'subject',
        'x_type': 'subject_type',
        'y': 'object',
        'y_type': 'object_type',
        'r': 'relation'
        }
        self.cls_task_only = cls_task_only
        self.output_dir = self.get_output_folder_name()
        self.preprompt = self.generate_preprompt()
        
        print("output_dir=",self.output_dir)

    def filter_skip_relations(self):
        return [r for r in self.all_relations if r not in self.allowed_relations]

    def get_output_folder_name(self):
        parts = [self.ds_root, f"{len(self.allowed_relations)}cls"]
        if self.cls_task_only:
            parts.append(f"clsTskOnl")
        if self.max_turns:
            parts.append(f"{self.max_turns}trn")
        if self.max_speakers:
            parts.append(f"{self.max_speakers}spkr")
        if self.skip_empty_triples:
            parts.append(f"skipEmptyPairs")
        if self.balance_empty_dialogues:
            parts.append(f"balPairs")
        if self.rebalance_empty_dialogues:
            parts.append(f"rebalPairs")
        if self.parse_subdialogues:
            parts.append(f"parseSubDlgs")
        if self.replace_skipped_with_others:
            parts.append(f"replSkpWthOth")
        if self.rewrite_keys:
            parts.append(f"rwrtKeys")
        if self.instruction_type != "A" and not self.cls_task_only:
            parts.append(f"instr{self.instruction_type}")
        if self.add_one_shot:
            parts.append(f"add1Sht")

        return os.path.join("/home/murilo/RelNetCare/data/processed", "-".join(parts))

    def replace_keys(self, text):
        for old, new in self.replacement_dict.items():
            text = text.replace(f'"{old}":', f'"{new}":')
        return text

    def get_instruction(self):
        preprompts = {
            "clsTskOnl": "Classify the relation between the source and object entities below, given the input dialogue.",
            "A": "Extract personal relevant entities, and their relations. Return only the jsonl format list.",
            "B": "Extract personal relevant entities, and their relations. Return only the jsonl format list. Extract entities and relations from the given dialogue input and generate a JSON list as output that is structured according to the entity and relation types from the ontology."
        }
        return preprompts[self.instruction_type]

    def generate_preprompt(self):
        
        one_shot = """
Input:
[
"Speaker 1: Emma got a cat, Max.",
"Speaker 2: Nice! Who is Emma?",
"Speaker 1: She's my sister.",
]

Output:
[
{"x": "Speaker 1", "x_type": "PERSON", "r": "siblings", "y": "Emma", "y_type": "PERSON"},
{"x": "Emma", "x_type": "PERSON", "r": "siblings", "y": "Speaker 1", "y_type": "PERSON"},
{"x": "Emma", "x_type": "PERSON", "r": "pet", "y": "Max", "y_type": "ANIMAL"},
{"x": "Max", "x_type": "ANIMAL", "r": "pet", "y": "Emma", "y_type": "PERSON"}
]
"""
        if self.rewrite_keys:
            one_shot = self.replace_keys(one_shot)

        preprompt = f"""
{self.get_instruction()}

Ontology: 
- relations: {str(self.allowed_relations).replace("'", '"')}
{'' if self.cls_task_only else '- types: {{"ORG", "GPE", "PERSON", "DATE", "EVENT", “ANIMAL”}}'}
{one_shot if self.add_one_shot else ""}
Input:
"""
        return preprompt
    
    
    
def get_config_from_stem(data_stem):
    kwargs = {}
    kwargs['cls_task_only'] = 'clsTskOnl' in data_stem
    kwargs['balance_empty_dialogues'] = 'balPairs' in data_stem
    kwargs['rebalance_empty_dialogues'] = 'rebalPairs' in data_stem
    kwargs['replace_skipped_with_others'] = 'replSkpWthOth' in data_stem
    kwargs['skip_empty_triples'] = 'skipEmptyPairs' in data_stem
    kwargs['parse_subdialogues'] = 'parseSubDlgs' in data_stem
    kwargs['rewrite_keys'] = 'rwrtKeys' in data_stem
    kwargs['add_one_shot'] = 'add1Sht' in data_stem

    class_count = int(re.search(r'(\d+)cls', data_stem).group(1))
    kwargs['ignore_relation_filter'] = class_count == 35 # max class count @TODO: improve this logic
    
    max_turns_match = re.search(r'(\d+)trn', data_stem)
    if max_turns_match:
        kwargs['max_turns'] = int(max_turns_match.group(1))
    
    max_speakers_match = re.search(r'(\d+)spkr', data_stem)
    if max_speakers_match:
        kwargs['max_speakers'] = int(max_speakers_match.group(1))
    
    instr_type_match = re.search(r'instr([A-Z])', data_stem)
    if instr_type_match:
        kwargs['instruction_type'] = instr_type_match.group(1)
    
    return LLMTransformationConfig(**kwargs)