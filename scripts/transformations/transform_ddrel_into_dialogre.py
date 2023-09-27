
import json
import os

# Mapping for 13-class to more granular relations
relation_map = {
    1:  "children", #Child-Parent
    2:  "children_others", #Child-Other Family Elder
    3:  "siblings", #Siblings
    4:  "spouse", #Spouse
    5:  "Lovers", 
    6:  "Courtship",
    7:  "Friends",
    8:  "Neighbors",
    9:  "Roommates",
    10: "Workplace Superior - Subordinate",
    11: "Colleague/Partners",
    12: "Opponents",
    13: "Professional Contact"
}

from src.paths import LOCAL_RAW_DATA_PATH, LOCAL_PROCESSED_DATA_PATH

# Directories
ddrel_dirs = [LOCAL_RAW_DATA_PATH / f"ddrel/{set_}.txt" for set_ in ('train', 'test', 'dev')]
dialogre_dirs = [LOCAL_RAW_DATA_PATH / f"dialog-re/{set_}.json" for set_ in ('train', 'test', 'dev')]
output_dirs = [LOCAL_PROCESSED_DATA_PATH / f"dialog-re-ddrel/{set_}.json" for set_ in ('train', 'test', 'dev')]

for txt_dir, json_dir, output_dir in zip(ddrel_dirs, dialogre_dirs, output_dirs):
    # Initialize the target list
    target_data = []

    # Read and transform the .txt data
    with open(txt_dir, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            
            # Extract relevant fields
            context = sample['context']
            label_13 = int(sample['label']) + 100  # Adding 100 to all IDs
            
            # Convert to target format
            dialogue = []
            relations = []
            nameA_found = False
            nameB_found = False
            prev_text = None  # To keep track of the previous text line

            for line in sample['context']:
                if ": " in line:
                    if prev_text:  # If there's text to append, add it to the dialogue
                        dialogue[-1] += " " + prev_text
                        prev_text = None  # Reset prev_text

                    speaker, text = line.split(": ", 1)
                    new_speaker = "Speaker 1" if speaker == 'A' else "Speaker 2"
                    dialogue.append(f"{new_speaker}: {text}")

                    if sample['nameA'] in text:
                        nameA_found = True
                    if sample['nameB'] in text:
                        nameB_found = True
                else:
                    # If the line doesn't contain ": ", append it to prev_text
                    if prev_text:
                        prev_text += " " + line
                    else:
                        prev_text = line
                
                if sample['nameA'] in text:
                    nameA_found = True
                if sample['nameB'] in text:
                    nameB_found = True

            if prev_text:  # If there's text left to append, add it to the dialogue
                dialogue[-1] += " " + prev_text

            relation = {
                "x": "Speaker 1",
                "x_type": "PER",
                "y": "Speaker 2",
                "y_type": "PER",
                "rid": [label_13],
                "r": [f"per:{relation_map[label_13 - 100]}"]  # Subtract 100 to get the original label
            }
            relations.append(relation)
            
            # Handle special cases
            if label_13 - 100 == 1:  # Child-Parent
                swapped_relation = {
                    "x": relation["y"],
                    "x_type": "PER",
                    "y": relation["x"],
                    "y_type": "PER",
                    "rid": [14],  # Set to 14 for parent relation
                    "r": ["per:parents"]
                }
                relations.append(swapped_relation)
            elif label_13 - 100 in [3, 4, 5,7,8,9,11,12,13]:  # Siblings, Spouse and others
                swapped_relation = {
                    "x": relation["y"],
                    "x_type": "PER",
                    "y": relation["x"],
                    "y_type": "PER",
                    "rid": [label_13],  # Keep the same rid
                    "r": [f"per:{relation_map[label_13 - 100]}"]
                }
                relations.append(swapped_relation)

            if nameA_found:
                relations.append({
                    "x": "Speaker 1",
                    "y": sample['nameA'],
                    "rid": [30],
                    "r": ["per:alternate_names"]
                })
            
            if nameB_found:
                relations.append({
                    "x": "Speaker 2",
                    "y": sample['nameB'],
                    "rid": [30],
                    "r": ["per:alternate_names"]
                })

            target_data.append([dialogue, relations])


    # Read the .json data
    with open(json_dir, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)

    # Merge the two datasets
    merged_data = existing_data + target_data

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # Save the merged data
    with open(output_dir, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)







# import json

# # Mapping for 13-class to more granular relations
# relation_map = {
#     1: "Child-Parent",
#     2: "Child-Other Family Elder",
#     3: "Siblings",
#     4: "Spouse",
#     5: "Lovers",
#     6: "Courtship",
#     7: "Friends",
#     8: "Neighbors",
#     9: "Roommates",
#     10: "Workplace Superior - Subordinate",
#     11: "Colleague/Partners",
#     12: "Opponents",
#     13: "Professional Contact"
# }

# # Initialize the target list
# target_data = []

# # Read the original data
# with open('/home/murilo/RelNetCare/data/raw/ddrel/dev.txt', 'r', encoding='utf-8') as f:
#     for line in f:
#         sample = json.loads(line.strip())
        
#         # Extract relevant fields
#         context = sample['context']
#         label_13 = int(sample['label']) + 100  # Adding 100 to all IDs
        
#         # Convert to target format
#         dialogue = []
#         relations = []
#         nameA_found = False
#         nameB_found = False
#         for line in context:
#             speaker, text = line.split(": ", 1)
#             new_speaker = "Speaker 1" if speaker == 'A' else "Speaker 2"
#             dialogue.append(f"{new_speaker}: {text}")

#             if sample['nameA'] in text:
#                 nameA_found = True
#             if sample['nameB'] in text:
#                 nameB_found = True
        
#         relation = {
#             "x": "Speaker 1",
#             "x_type": "PER",
#             "y": "Speaker 2",
#             "y_type": "PER",
#             "rid": [label_13],
#             "r": [relation_map[label_13 - 100]]  # Subtract 100 to get the original label
#         }
#         relations.append(relation)

#         if nameA_found:
#             relations.append({
#                 "x": "Speaker 1",
#                 "y": sample['nameA'],
#                 "rid": [30],
#                 "r": ["per:alternate_names"]
#             })
        
#         if nameB_found:
#             relations.append({
#                 "x": "Speaker 2",
#                 "y": sample['nameB'],
#                 "rid": [30],
#                 "r": ["per:alternate_names"]
#             })

#         target_data.append([dialogue, relations])

# # Save the target data
# with open('converted_data.json', 'w', encoding='utf-8') as f:
#     json.dump(target_data, f, ensure_ascii=False, indent=4)
