# dump read_me
import pandas as pd
import json
import os




def format_relation(x):
    try:
        return f"{x['rid'][0]:02}__{x['r'][0]}"
    except TypeError as e:
        print(f"TypeError for item {x}: {e}")
        return None  # or some default value



def dump_readme(data_path):
    files = os.listdir(data_path)
    data_frames = []

    files = [f for f in files if f != "relation_label_dict.json" and f.endswith('.json')]

    for file in files:
        with open(os.path.join(data_path, file), 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data, columns=['dialog', 'relation_info'])
        df['dataset_name'] = file.replace('.json', '')
        data_frames.append(df)

    final_df = pd.concat(data_frames, ignore_index=True)
    x_df = final_df.explode('relation_info')
    x_df['r'] = x_df['relation_info'].apply(format_relation)

    # Create a pivot table
    pivot_df = pd.pivot_table(x_df, values='relation_info', index=['r'], columns=['dataset_name'], aggfunc='count', fill_value=0)

    # Rename the columns and calculate the proportion and total
    pivot_df.columns = [f'count_{col}' for col in pivot_df.columns]
    pivot_df['total'] = pivot_df.sum(axis=1)
    pivot_df['proportion'] = (pivot_df['total'] / pivot_df['total'].sum() * 100).apply(lambda x: f"{x:.1f}%")

    # Split 'r' into 'rid' and 'relation'
    pivot_df.reset_index(inplace=True)
    pivot_df[['rid', 'r']] = pivot_df['r'].str.split('__', expand=True)
    pivot_df = pivot_df.sort_values('total', ascending=False)
    
    readme_text = f"""
# Data Descriptions

## Dataset Size 

### Dialogue Counts

{ final_df.dataset_name.value_counts().to_markdown() }

### Relation Counts

{ x_df.dataset_name.value_counts().to_markdown() }

## Relationship Types (Relation Counts)

{pivot_df.set_index('rid').to_markdown()}
    """

    with open(os.path.join(data_path, "README.md"), "w") as f:
        f.write(readme_text)



def filter_relations(data, allowed_relations):
	"""Filter relations in the dataset."""
	for item in data:
		filtered_relations = [relation for relation in item[1] if relation['r'][0] in allowed_relations]
		item[1] = filtered_relations
	return data

def modify_relation_names_and_ids(data):
    for item in data:
        for relation in item[1]:
            relation_name = relation['r'][0]
            if 'no_relation' in relation_name or 'unanswerable' in relation_name:
                relation['r'] = ['no_relation']
            else:
                relation['r'] = ['with_relation']
    return data

def create_continuous_mapping(allowed_relations, make_binary=False, force_original=False):
    """Create a continuous mapping for the allowed relations."""
    if make_binary:
        continuous_mapping =  { "1": 'no_relation', "2": "with_relation"}
    else:
        continuous_mapping = {str(idx + 1): relation for idx, relation in enumerate(allowed_relations)}
        
    if force_original:
        continuous_mapping = {
            "1": "per:positive_impression",
            "2": "per:negative_impression",
            "3": "per:acquaintance",
            "4": "per:alumni",
            "5": "per:boss",
            "6": "per:subordinate",
            "7": "per:client",
            "8": "per:dates",
            "9": "per:friends",
            "10": "per:girl_boyfriend",
            "11": "per:neighbor",
            "12": "per:roommate",
            "13": "per:children",
            "14": "per:other_family",
            "15": "per:parents",
            "16": "per:siblings",
            "17": "per:spouse",
            "18": "per:place_of_residence",
            "19": "per:place_of_birth",
            "20": "per:visited_place",
            "21": "per:origin",
            "22": "per:employee_or_member_of",
            "23": "per:schools_attended",
            "24": "per:works",
            "25": "per:age",
            "26": "per:date_of_birth",
            "27": "per:major",
            "28": "per:place_of_work",
            "29": "per:title",
            "30": "per:alternate_names",
            "31": "per:pet",
            "32": "gpe:residents_of_place",
            "33": "per:age",
            "34": "gpe:visitors_of_place",
            "35": "org:employees_or_members",
            "36": "org:students",
            "37": "no_relation"
        }
    
    # Create the reverse mapping for data update
    reverse_mapping = {v: k for k, v in continuous_mapping.items()}
    
    return continuous_mapping, reverse_mapping

def update_data_rid(data, reverse_mapping):
    """Update the rid in the data to match the new continuous mapping."""
    for item in data:
        for relation in item[1]:
            relation_name = relation['r'][0]
            if relation_name in reverse_mapping: # Add this check
                new_rid = reverse_mapping[relation_name]
                relation['rid'] = [int(new_rid)]
    return data

def create_new_mapping(data, allowed_relations):
	"""Create a new mapping based on the rid values for the allowed relations."""
	rid_relation_mapping = {}
	for item in data:
		for relation in item[1]:
			if relation['r'][0] in allowed_relations:
				rid = relation['rid'][0]
				rname = relation['r'][0]
				rid_relation_mapping[str(rid)] = rname
	return rid_relation_mapping



def process_files(raw_data_dir, processed_dir, allowed_relations, make_binary=False, filter_data=True):
    if make_binary:
        output_dir = f"{raw_data_dir}-binary"
    else:
        output_dir = f"{processed_dir}/dialog-re-{len(allowed_relations)}cls"   
    
    
    os.makedirs(output_dir, exist_ok=True)
    # Load the training data to create the mapping
    with open(os.path.join(raw_data_dir, "train.json"), 'r') as file:
        train_data = json.load(file)
    
    # Create continuous and reverse mappings using the training data
    continuous_mapping, reverse_mapping = create_continuous_mapping(allowed_relations, make_binary=make_binary)
    print(continuous_mapping)

    # Process all data files
    for file_name in os.listdir(raw_data_dir):
        if file_name.endswith('.json') and file_name != 'relation_label_dict.json':
            with open(os.path.join(raw_data_dir, file_name), 'r') as file:
                data = json.load(file)
                
            if make_binary:
                data = modify_relation_names_and_ids(data)
            
            # Update the rid values in the data
            updated_data = update_data_rid(data, reverse_mapping)
            if filter_data:
                updated_data = filter_relations(updated_data, allowed_relations)

            # Save the filtered data
            output_path = f"{output_dir}/{file_name}"
            with open(output_path, 'w') as file:
                json.dump(updated_data, file)
                
    # Save the new mapping
    dump_path = os.path.join(output_dir, "relation_label_dict.json")
    with open(dump_path, 'w') as file:
        json.dump(continuous_mapping, file)

    print("output_dir=", output_dir)
    print("Done processing files!")
    return output_dir


# Path to the raw data directory and processed directory
# raw_data_dir = "/home/murilo/RelNetCare/data/raw/dialog-re"
if __name__ == "__main__":
    # Allowed relations
    allowed_relations = [
        "per:other_family",
        "per:children",
        "per:parents", 
        "per:siblings",
        "per:spouse",
        "gpe:visitors_of_place",

        "per:acquaintance",
        "per:pet", 
        "per:place_of_residence",
        "per:visited_place", 
        "gpe:residents_of_place",
        
        # "no_relation",
        
        # "per:alternate_names",
        # "per:girl_boyfriend",
        # "per:positive_impression",
        # "per:friends",
        # "per:title",
        # "per:spouse",
        # "per:siblings",
        # "per:parents",
        # "per:children",
        # "per:negative_impression",
        # "per:roommate",
        # "per:alumni",
        # "per:other_family",
        # "per:visited_place",
        # "gpe:visitors_of_place",
        # "per:works",
        # "per:client",
        # "per:place_of_residence",
        # "gpe:residents_of_place",
        # "per:age",
        # "per:boss",
        # "org:employees_or_members",
        # "per:employee_or_member_of",
        # "per:place_of_work",
        # "per:acquaintance",
        # "per:subordinate",
        # "per:dates",
        # "per:neighbor",
        # "per:pet",
        # "per:origin",
        # "org:students",
        # "per:schools_attended",
        # "per:date_of_birth",
        # "per:major",
        # "per:place_of_birth",
        # "gpe:births_in_place",
        ]


    raw_data_dir  = "/home/murilo/RelNetCare/data/processed/dialog-re-37cls-with-no-relation-undersampled"
    processed_dir = "/home/murilo/RelNetCare/data/processed"
    output_dir = process_files(raw_data_dir, processed_dir, allowed_relations)
    dump_readme(output_dir)
    print("output_dir=",output_dir)