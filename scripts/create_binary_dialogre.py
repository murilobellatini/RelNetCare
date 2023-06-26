from src.paths import LOCAL_PROCESSED_DATA_PATH
from src.statistics import get_counts_and_percentages

from pathlib import Path
import pandas as pd
import json
import glob
import math
import os

files = [Path(f) for f in glob.glob(str(LOCAL_PROCESSED_DATA_PATH / "dialog-re-fixed-relations/*.json")) if 'relation_label_dict.json' not in str(f)]

# Define new directory for the output files
new_dir = LOCAL_PROCESSED_DATA_PATH / "dialog-re-ternary"

# Create directory if it doesn't exist
os.makedirs(new_dir, exist_ok=True)

# Define a function to overwrite the relation types
def overwrite_relations(data):
    for item in data:
        # item[1] corresponds to the list of relations
        for rel in item[1]:
            # Check if the relation type is 'no_relation'
            if rel['r'][0] == 'no_relation':
                rel['rid'][0] = 0  # Set 'rid' to 0 for 'no_relation'
            # Check if the relation type is 'unanswerable'
            elif rel['r'][0] == 'unanswerable':
                rel['rid'][0] = 1  # Set 'rid' to 2 for 'unanswerable'
            else:
                rel['r'][0] = "with_relation" 
                rel['rid'][0] = 2  # Set 'rid' to 1 for 'with_relation'
    return data


# Loop over all files
for file in files:
    with open(file, 'r') as json_file:
        data = json.load(json_file)
        
    # Overwrite relations
    data = overwrite_relations(data)
    
    # Determine the set (train, dev, test) based on the filename
    set_type = file.stem.split('_')[-1]  # This assumes that the set type is always at the end of the file name
    
    # Write back to a new JSON file
    with open(new_dir / f"{set_type}.json", 'w') as json_file:
        json.dump(data, json_file)
