import re
import os
import torch
import stringcase
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def format_dir(input_data_dir):
    words_after_dash = re.findall('dialog-re-(.*)', input_data_dir)[0]
    words = words_after_dash.split('-')
    camel_case = ''.join(word.capitalize() for word in words)
    no_vowels = re.sub('[aeiou]', '', camel_case)
    return no_vowels


class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'

class LLMTransformationConfig:
    def __init__(self,
                 max_turns=None,
                 max_speakers=None,
                 cls_task_only=False,
                 triplet_to_text=True,
                 skip_types=False,
                 max_turn_cap=None,
                 balance_empty_dialogues=False,
                 rebalance_empty_dialogues=False,
                 replace_skipped_with_others=False,
                 skip_empty_triples=False,
                 parse_subdialogues=False,
                 rewrite_keys=True,
                 add_one_shot=False,
                 instruction_type="A",
                 ignore_relation_filter=False,
                 rebalance_multiplier=1,
                 group_classes=None,
                 shuffle_data=False,
                 input_data_dir=None,
                 merge_places=False,
                 file_sets = [['train', 'dev'], ['test']]
                 ):
        
        self.grouped_relations = {
                "Attachment": ["roommate", "pet", "client", "dates", "other_family", 
                            "children", "parents", "acquaintance", "spouse", "friends", 
                            "girl/boyfriend", "siblings"],

                "Identity": ["date_of_birth", "title", "major", "origin", "place_of_birth",
                            "births_in_place", "age", "alternate_names"],

                "Comfort": ["negative_impression", "positive_impression"],

                "Occupation": ["place_of_work", "employees_or_members", "subordinate", 
                            "boss", "works", "students", "schools_attended", "alumni", 
                            "employee_or_member_of"],

                "Inclusion": ["neighbor", "place_of_residence", "residents_of_place",
                            "visitors_of_place", "visited_place"],
                
                "Others" : ['unanswerable', 'no_relation'],
                "DDRel" : [
                    "children", #Child-Parent
                    "children_others", #Child-Other Family Elder
                    "siblings", #Siblings
                    "spouse", #Spouse
                    "Lovers",
                    "Courtship",
                    "Friends",
                    "Neighbors",
                    "Roommates",
                    "Workplace Superior - Subordinate",
                    "Colleague/Partners",
                    "Opponents",
                    "Professional Contact"
                ]
            }

        self.all_relations = set().union(*self.grouped_relations.values())

        # Adhoc way to filter relations allowed @todo: create logic 
        # self.allowed_relations = {
        #     "place_of_residence", "visited_place", "residents_of_place", "visitors_of_place"
        #     }
        self.allowed_relations = {
            "acquaintance",  "other_family", "pet", 
            "children", "parents", 
            "siblings", "spouse", "place_of_residence", "visited_place", 
            "residents_of_place", "visitors_of_place"
            }
        # self.allowed_relations = deepcopy(self.all_relations)
        
        # if cls_task_only:
        #     self.all_relations.add('inverse_relation')
        #     self.all_relations.add('births_in_place') #single sample
        #     self.all_relations.add('place_of_birth') #single sample
        #     self.allowed_relations.remove('births_in_place')
        #     self.allowed_relations.remove('place_of_birth')
            
        if group_classes:
            self.allowed_relations = set()
            for cls in group_classes:
                if cls in self.grouped_relations:
                    self.allowed_relations.update(self.grouped_relations[cls])
            
        self.ignore_relation_filter = ignore_relation_filter
        
        if self.ignore_relation_filter:
            if group_classes:
                self.allowed_relations = self.group_classes
            else:
                self.allowed_relations = self.all_relations
            
        
        self.skip_relations = self.filter_skip_relations()
        self.total_relation_count = len(self.skip_relations)

        self.merge_places = merge_places
        
        if self.merge_places:
            self.allowed_relations.remove('place_of_residence') 
            self.allowed_relations.remove('residents_of_place') 
            self.allowed_relations.remove('visitors_of_place') 

        self.max_turn_cap = max_turn_cap 
        self.simpler_types = True
        self.triplet_to_text = triplet_to_text
        self.ds_type = ""
        if not input_data_dir:
            self.input_dir = "/home/murilo/RelNetCare/data/raw/dialog-re"
            self.default_data_dir = True
        else: 
            self.input_dir = input_data_dir
            self.default_data_dir = False
        self.ds_root = f"{self.input_dir.split('/')[-1]}-llama{self.ds_type}"
        self.file_sets = file_sets
        self.max_turns = max_turns
        self.max_speakers = max_speakers
        self.skip_empty_triples = skip_empty_triples
        self.balance_empty_dialogues = balance_empty_dialogues
        self.rebalance_empty_dialogues = rebalance_empty_dialogues
        self.parse_subdialogues = parse_subdialogues
        self.replace_skipped_with_others = replace_skipped_with_others
        self.rewrite_keys = False if cls_task_only else rewrite_keys
        self.add_one_shot = False if cls_task_only else add_one_shot
        self.rebalance_multiplier = rebalance_multiplier
        self.shuffle_data = shuffle_data
        self.group_classes = group_classes
        if cls_task_only:
            self.instruction_type = 'clsTskOnl' + (f'{instruction_type}' if instruction_type != 'A' else '')
        elif triplet_to_text:
            self.instruction_type = f'trToDial{instruction_type}'
        else:
            self.instruction_type = instruction_type
        self.replacement_dict = {
        'x': 'subject',
        'x_type': 'subject_type',
        'y': 'object',
        'y_type': 'object_type',
        'r': 'relation'
        }
        self.cls_task_only = cls_task_only
        self.skip_types = skip_types
        if self.skip_types:
            print(f'self.skip_types={self.skip_types}, forcing intruction_type to `C`...')
            self.instruction_type = 'C'
        self.output_dir = self.get_output_folder_name()
        self.preprompt = self.generate_preprompt()
        
        print("output_dir=",self.output_dir)

    def filter_skip_relations(self):
        return [r for r in self.all_relations if r not in self.allowed_relations]

    def get_output_folder_name(self):
        parts = [self.ds_root, f"{len(self.allowed_relations)}cls"]
        if self.cls_task_only:
            parts.append(f"clsTskOnl")
        if self.triplet_to_text:
            parts.append(f"trToDial")            
        if self.max_turns:
            parts.append(f"{self.max_turns}trn")
        if self.max_speakers:
            parts.append(f"{self.max_speakers}spkr")
        if self.skip_empty_triples:
            parts.append(f"skipEmptyPairs")
        if self.balance_empty_dialogues:
            parts.append(f"balPairs")
        if self.rebalance_empty_dialogues:
            ext = f"{self.rebalance_multiplier}x" if self.rebalance_multiplier != 1 else ''
            parts.append(f"rebalPairs{ext}")
        if self.parse_subdialogues:
            parts.append(f"parseSubDlgs")
        if self.replace_skipped_with_others:
            parts.append(f"replSkpWthOth")
        if self.rewrite_keys:
            parts.append(f"rwrtKeys")
        if self.instruction_type != "A":
            parts.append(f"instr{self.instruction_type[-1]}")
        if self.add_one_shot:
            parts.append(f"add1Sht")
        if self.max_turn_cap:
            parts.append(f"mxTrnCp{self.max_turn_cap}")
        if self.shuffle_data:
            parts.append(f"shfflDt")
        if self.skip_types:
            parts.append(f"skpTps")        
        if self.merge_places:
            parts.append(f"mrgPlcs")        
        if self.group_classes:
            parts.append(f"GrpCls{''.join(self.group_classes)}")
        if self.input_dir != "/home/murilo/RelNetCare/data/raw/dialog-re":
            parts.append(format_dir(self.input_dir))
            
        return os.path.join("/home/murilo/RelNetCare/data/processed", "-".join(parts))

    def replace_keys(self, text):
        for old, new in self.replacement_dict.items():
            text = text.replace(f'"{old}":', f'"{new}":')
        return text

    def get_template(self):
        templates = {
            "clsTskOnl": "Classify the relation between the source and object entities below, given the input dialogue.\n{one_shot}\nOntology:\n{ontology}{types}\nInput: {input_dialogue}\n\nSubject: {input_subject}\nObject: {input_object}\nRelation:",
            "clsTskOnlB": "Ontology:\n{ontology}{types}{one_shot}\n\nInput Dialogue: {input_dialogue}\n\nSubject: {input_subject}\nObject: {input_object}\nRelation: Pick one ontology label describing the subject-object link. Only the label.",
            "A": "Extract personal relevant entities, and their relations. Return only the jsonl format list.\n{one_shot}\nOntology:\n{ontology}{types}\n\nInput: {input_dialogue}\n\nOutput:",
            "B": "Extract personal relevant entities, and their relations. Return only the jsonl format list. Extract entities and relations from the given dialogue input and generate a JSON list as output that is structured according to the entity and relation types from the ontology.\n{one_shot}\nOntology:\n{ontology}{types}\n\nInput: {input_dialogue}\n\nOutput:",
            "C": 'Extract entities and relations from the dialogue. Return a Python list of JSON objects, each fitting this schema: {{"subject": "<Entity>", "subject_type": "<{slashed_types}>", "relation": "<{slashed_ontology}>", "object": "<Related Entity>", "object_type": "<{slashed_types}>"}}. No additional text or explanations. Return an empty list if no relevant entities or relations are found. Stick to the provided types and relations. You are like an API, you don\'t speak you only return JSON objects.\nDialogue: {input_dialogue}',
            "trToDialA": "Your task is to write a {turn_count} turn dialogue based on the following relationships:\nInput Relations: {input_relations}\nOutput Dialogue: Please write the dialogue in a Python list format, like this: ['turn1', 'turn2']. Only return the list and nothing else.\n",
            "trToDialB": "Your task is to write a {turn_count} turn dialogue based on the following relationships:\nInput Relations: {input_relations}\nOutput Dialogue: Ensure that the generated output only includes the provided information from the triples. Please write the dialogue in a Python list format, like this: ['turn1', 'turn2']. Only return the list and nothing else.\n"
        }
        if self.skip_types:
            templates['C'] = 'Extract entities and relations from the dialogue. Return a Python list of JSON objects, each fitting this schema: {{"subject": "<Entity>", "relation": "<{slashed_ontology}>", "object": "<Related Entity>"}}. No additional text or explanations. Return an empty list if no relevant entities or relations are found. Stick to the provided relations. You are like an API, you don\'t speak you only return JSON objects.\nDialogue: {input_dialogue}'
        return templates[self.instruction_type]

    def get_one_shot(self):
        if self.cls_task_only:
            one_shot = """
Example Input:
[
"Speaker 1: Emma got a cat, Max.",
"Speaker 2: Nice! Who is Emma?",
"Speaker 1: She's my sister.",
]

Example Subject: Speaker 1
Example Object: Emma
Example Relation: siblings
"""
        else:
            
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
        
        return one_shot
            
    def generate_preprompt(self):
        if self.simpler_types:
            types = {"Organization", "Location", "Person", "Date", "Event", "Animal"}
        else:
            types = {"ORG", "GPE", "PERSON", "DATE", "EVENT", "ANIMAL"}
            
        if self.group_classes:
            relations = self.group_classes
        else:
            relations = self.allowed_relations
            
        variables = SafeDict({
            'ontology': f"- Relations: {str(sorted(relations))}".replace("'", '"'),
            'slashed_ontology': "/".join(sorted(relations)),
            'types': '' if self.cls_task_only else f'\n- Types: {types}\n',
            'slashed_types': "/".join(types),
            'one_shot': self.get_one_shot() if self.add_one_shot else '',
        })

        # Filling the placeholders
        preprompt = self.get_template().format_map(variables)

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
    kwargs['triplet_to_text'] = 'trToDial' in data_stem
    kwargs['shuffle_data'] = 'shfflDt' in data_stem
    kwargs['skip_types'] = 'skpTps' in data_stem
    kwargs['merge_places'] = 'mrgPlcs' in data_stem

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
    
    max_turn_cap_match = re.search(r'mxTrnCp([A-Z])', data_stem)
    if max_turn_cap_match:
        kwargs['max_turn_cap'] = max_turn_cap_match.group(1)
   
    rebal_mult_match = re.search(r'rebalPairs([\d.]+)x', data_stem)
    if max_turn_cap_match:
        kwargs['rebalance_multiplier'] = rebal_mult_match.group(1)     
        
    group_classes_match = re.search(r'GrpCls', data_stem)
    if group_classes_match:
        # Remove the 'GrpCls' part
        group_classes_stem = re.sub('GrpCls', '', data_stem)
        # Split based on capital letters
        kwargs['group_classes'] = re.findall('[A-Z][^A-Z]*', group_classes_stem)
        
    return LLMTransformationConfig(**kwargs)