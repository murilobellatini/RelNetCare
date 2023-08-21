import os
import json

def filter_skip_relations(all_relations, allowed_relations):
    """Return a list of relations not in allowed_relations."""
    return [r for r in all_relations if r not in allowed_relations]


def transform_data_to_llm_format(data, preprompt="", start_idx=0, skip_relations=[], max_turns=10, max_speakers=2):
    """Transform the given data into LLM ready dialogue format."""
    new_data = []
    identity_counter = start_idx

    for conversation, triples in data:
        # Filtering dialogues based on max_turns criteria
        if max_turns is not None and len(conversation) > max_turns:
            continue

        # Extracting speaker names from each dialogue turn and checking based on max_speakers criteria
        if max_speakers is not None:
            speakers = set([turn.split(":")[0].strip() for turn in conversation])
            if len(speakers) > max_speakers:
                continue


        triples_text = [
            {
                "x": triple["x"],
                "x_type": triple["x_type"],
                "r": triple["r"][0].split(':')[-1],
                "y": triple["y"],
                "y_type": triple["y_type"]
            }
            for triple in triples
            if triple["r"] and triple["r"][0].split(':')[-1] not in skip_relations
        ]

        conversation_entry = {
            "id": f"identity_{identity_counter}",
            "conversations": [
                {"from": "human", #tried: human
                 "value": preprompt + "\n".join(conversation)},
                {"from": "gpt", #tried: gpt
                 "value": str(json.dumps(triples_text))}
            ]
        }

        identity_counter += 1

        if triples_text:
            new_data.append(conversation_entry)

    return new_data


def create_directory_if_not_exists(directory_path):
    """Create a directory if it does not already exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def process_and_save_data(file_sets, input_dir, output_dir, skip_relations, preprompt="", total_relation_count=35, max_turns=10, max_speakers=2):
    """Process the input data and save in the specified format."""
    output_data = []
    for files in file_sets:
        data_folder_name = output_dir.split('/')[-1]
        dataset_name = f"{data_folder_name}-{total_relation_count-len(skip_relations)}cls-{'-'.join(files)}"
        last_data_idx = 0
        new_format = []

        create_directory_if_not_exists(output_dir)

        for f in files:
            input_data_path = os.path.join(input_dir, f'{f}.json')
            
            # Check if file exists
            if os.path.exists(input_data_path):
                with open(input_data_path, encoding='utf8') as fp:
                    data = json.load(fp)

                new_format.extend(transform_data_to_llm_format(
                    data, preprompt=preprompt, start_idx=last_data_idx,
                    skip_relations=skip_relations, max_turns=max_turns, max_speakers=max_speakers))
                last_data_idx = len(new_format)

        output_data_path = os.path.join(output_dir, f'{dataset_name}.json')
        with open(output_data_path, 'w', encoding='utf8') as fp:
            json.dump(new_format, fp)

        print(files, len(new_format))
        
        output_data.append(new_format)
        
    return output_data
        
add_one_shot = False
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

all_relations = {
"positive_impression", "negative_impression", "acquaintance", 
"alumni", "boss", "subordinate", "client", "dates", "friends", 
"girl/boyfriend", "neighbor", "roommate", "children", "other_family", 
"parents", "siblings", "spouse", "place_of_residence", "visited_place", 
"origin", "employee_or_member_of", "schools_attended", "works", "age", 
"date_of_birth", "major", "place_of_work", "title", "alternate_names", 
"pet", "residents_of_place", "visitors_of_place", "employees_or_members", 
"students", "unanswerable"
}
allowed_relations = {"acquaintance", "children", "other_family", "parents", 
                    "siblings", "spouse", "place_of_residence", "visited_place", 
                    "pet", "residents_of_place", "visitors_of_place"}

# allowed_relations = all_relations  # uncomment to allow all relations!

preprompt = f"""
Extract personal relevant entities, and their relations. Return only the jsonl format list .

Ontology: 
- relations: {str(allowed_relations).replace("'", '"')}
- types: {{"ORG", "GPE", "PERSON", "DATE", "EVENT", “ANIMAL”}}
{one_shot if add_one_shot else ""}
Input:
"""


def get_output_folder_name(ds_root, skip_relations, max_turns=10, max_speakers=2):
    """Generate a concise and descriptive output folder name."""
    parts = [ds_root, f"{len(all_relations) - len(skip_relations)}cls"]
    
    if max_turns:
        parts.append(f"{max_turns}trn")
    
    if max_speakers:
        parts.append(f"{max_speakers}spkr")
    
    return "-".join(parts)

skip_relations = filter_skip_relations(all_relations, allowed_relations)

# Directories
ds_type = ""
# ds_type = "-typed-pp"
max_turns=10
max_speakers=2
ds_root = f"dialog-re-llama{ds_type}"
INPUT_DIR = "/home/murilo/RelNetCare/data/raw/dialog-re"
OUTPUT_DIR = os.path.join("/home/murilo/RelNetCare/data/processed", get_output_folder_name(
    ds_root=ds_root, skip_relations=skip_relations, max_turns=max_turns, max_speakers=max_speakers))

FILE_SETS = [['train', 'dev'], ['test']]

print(preprompt)

data = process_and_save_data(FILE_SETS, INPUT_DIR, OUTPUT_DIR, skip_relations, preprompt, max_turns=max_turns, max_speakers=max_speakers)