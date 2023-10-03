import os
import json
import re
from ast import literal_eval

class RelationConverter:
    
    def __init__(self, input_path):
        self.input_path = input_path
        self.output_path = input_path + '-prepBART'
        print(f"Initialized with input path: {self.input_path}")
        print(f"Output path set to         : {self.output_path}")

    def _convert_relation_to_sentence(self, subject, relation, obj):
        if relation == 'visitors_of_place':
            return f"{subject} has visitors like {obj}"
        elif relation == 'spouse':
            return f"{subject}'s spouse is {obj}"
        elif relation == 'visited_place':
            return f"{subject} has visited {obj}"
        elif relation == 'siblings':
            return f"{subject} is a sibling of {obj}"
        elif relation == 'residents_of_place':
            return f"{subject} has residents like {obj}"
        elif relation == 'pet':
            return f"{subject} has a pet named {obj}"
        elif relation == 'place_of_residence':
            return f"{subject}'s place of residence is {obj}"
        elif relation == 'parents':
            return f"{subject} is a parent of {obj}"
        elif relation == 'children':
            return f"{subject} is a child of {obj}"
        else:
            return f"{subject} is {relation.replace('_', ' ')} of {obj}"
            
    def process_llama_json_to_bart_sentence(self):
        print(f"Processing JSON files from: {self.input_path} to: {self.output_path}")
        # Create the output directory if it doesn't exist
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            print(f"Created output directory: {self.output_path}")
        
        # Your original loop for reading and processing JSON files
        for filename in os.listdir(self.input_path):
            if filename.endswith('.json'):
                print(f"Processing file: {filename}")
                with open(os.path.join(self.input_path, filename), 'r') as f_in:
                    data = json.load(f_in)
                
                processed_data = []
                
                for entry in data:
                    processed_entry = self._process_single_entry(entry)
                    processed_data.append(processed_entry)
                
                matches = re.findall(r'(train|test|dev)', filename)
                filename_shorter = '-'.join(matches) + '.json'
                with open(os.path.join(self.output_path, filename_shorter), 'w') as f_out:
                    json.dump(processed_data, f_out, indent=4)
                    
    def _process_single_entry(self, entry):
        processed_entry = {'id': entry['id']}
        dialogue_list = literal_eval(entry['conversations'][0]['value'].split("Dialogue: ")[1])
        processed_entry['input'] = ' '.join(dialogue_list)
        
        relations = json.loads(entry['conversations'][1]['value'])
        output_value = []
        for rel in relations:
            output_str = self._convert_relation_to_sentence(rel['subject'], rel['relation'], rel['object'])
            output_value.append(output_str)
        
        processed_entry['output'] = '. '.join(output_value) + '.' if output_value else ''
        return processed_entry
    
    def _convert_sentence_to_relation(self, sentence):
        if "has visitors like" in sentence:
            subject, obj = sentence.split(" has visitors like ")
            return subject, 'visitors_of_place', obj
        elif "'s spouse is " in sentence:
            subject, obj = sentence.split("'s spouse is ")
            return subject, 'spouse', obj
        elif " has visited " in sentence:
            subject, obj = sentence.split(" has visited ")
            return subject, 'visited_place', obj
        elif " is a sibling of " in sentence:
            subject, obj = sentence.split(" is a sibling of ")
            return subject, 'siblings', obj
        elif " has residents like " in sentence:
            subject, obj = sentence.split(" has residents like ")
            return subject, 'residents_of_place', obj
        elif " has a pet named " in sentence:
            subject, obj = sentence.split(" has a pet named ")
            return subject, 'pet', obj
        elif "'s place of residence is " in sentence:
            subject, obj = sentence.split("'s place of residence is ")
            return subject, 'place_of_residence', obj
        elif " is a parent of " in sentence:
            subject, obj = sentence.split(" is a parent of ")
            return subject, 'parents', obj
        elif " is a child of " in sentence:
            subject, obj = sentence.split(" is a child of ")
            return subject, 'children', obj
        elif " is acquaintance of " in sentence:
            subject, obj = sentence.split(" is acquaintance of ")
            return subject, 'acquaintance', obj
        else:
            subject, relation, obj = re.split(' is | of ', sentence)
            return subject, relation.replace(' ', '_'), obj
    
    def test_conversion(self):
        print("Running test conversions...")
        # Test the functions
        test_sentences = [
            "Westchester has visitors like Speaker 1",
            "Speaker 1's spouse is Speaker 2",
            "Speaker 1 has visited Westchester",
            "Speaker 2 is a sibling of Frank",
            "Speaker 2 has residents like Boston",
            "Speaker 2 has a pet named Buddy",
            "Speaker 2's place of residence is Boston",
            "Emma is a parent of Speaker 2",
            "Speaker 1 is other family of Ben",
            "Speaker 2 is a child of Emma",
            "Speaker 1 is acquaintance of Speaker 2",
            
        ]

        for sentence in test_sentences:
            subject, relation, obj = self._convert_sentence_to_relation(sentence)
            print(f"Sentence: {sentence}")
            print(f"Subject: {subject}, Relation: {relation}, Object: {obj}\n")