import os
import json
import copy
import glob
import shutil
import itertools
import pandas as pd
from pathlib import Path
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from src.processing.text_preprocessing import DialogueEnricher
from src.paths import LOCAL_RAW_DATA_PATH, LOCAL_PROCESSED_DATA_PATH


class DialogREDatasetTransformer:
    """
    A utility for modifying DialogRE datasets, emphasizing cases without relations. Key methods:

    `add_no_relation_labels`: Adds "no relation" instances to the dataset.

    `transform_to_binary`: Transforms the dataset to express "no relation", "unanswerable", or "with relation".
    
    `transform_to_ternary`: Transforms the dataset into a binary version, merging "unanswerable" with "no relation",
                            and renaming all other relations as "with relation".
    """
    
    def __init__(self, raw_data_folder=LOCAL_RAW_DATA_PATH / 'dialog-re/'):
        self.raw_data_folder = raw_data_folder
        self.df = pd.DataFrame(columns=["Dialogue", "Relations", "Origin"])

    def load_data_to_dataframe(self):
        # Get a list of all json files in the directory, excluding 'relation_label_dict'
        files = [Path(f) for f in glob.glob(f"{self.raw_data_folder}/*.json") if "relation_label_dict" not in str(f)]

        # Loop over all json files in the directory
        for file_name in files:
            with open(file_name, 'r', encoding='utf8') as file:
                data = json.load(file)

                # Convert the data to a DataFrame
                df_temp = pd.DataFrame(data, columns=["Dialogue", "Relations"])

                # Add a new column to this DataFrame for the origin
                df_temp["Origin"] = file_name.stem  # This will get just the file name without the extension

                # Append the temporary DataFrame to the main DataFrame
                self.df = pd.concat([self.df, df_temp], ignore_index=True)

        return self.df

    def _load_data(self, file_path):
        with open(file_path, 'r', encoding='utf8') as file:
            data = json.load(file)
        return data

    def _find_relation_pairs(self, data):
        relation_pairs = set()
        for r in data[1]:  
            x = f"{r['x']}_{r['x_type']}"
            y = f"{r['y']}_{r['y_type']}"
            relation_pairs.add((x, y))
        return relation_pairs

    def _find_all_relation_permutations(self, relation_pairs):
        unique_items = set(item for sublist in relation_pairs for item in sublist)
        unique_combinations = set(itertools.permutations(unique_items, 2))
        return unique_combinations

    def _exclude_existing_relations(self, data, relation_pairs):
        existing_relations = set()
        for relation in data[1]:
            x = f"{relation['x']}_{relation['x_type']}"
            y = f"{relation['y']}_{relation['y_type']}"
            existing_relations.add((x, y))
        new_relations = relation_pairs - existing_relations
        return new_relations

    def _create_new_dialogues_with_new_relations(self, data, all_new_relations):
        new_dialogues = []
        for i, dialogue in enumerate(data):
            new_dialogue = copy.deepcopy(dialogue)
            relation_pairs = self._find_relation_pairs(dialogue)  # get the existing relation pairs
            for relation_pair in all_new_relations[i]:
                x, x_type = relation_pair[0].split('_')
                y, y_type = relation_pair[1].split('_')
                inverse_pair = (f"{y}_{y_type}", f"{x}_{x_type}")
                if inverse_pair in relation_pairs:  # check if the inverse pair exists
                    new_relation = {
                        'y': y,
                        'x': x,
                        'rid': [39],
                        'r': ['inverse_relation'],
                        't': [''],
                        'x_type': x_type,
                        'y_type': y_type
                    }
                else:
                    new_relation = {
                        'y': y,
                        'x': x,
                        'rid': [38],  # set to 38 for 'no_relation'
                        'r': ['no_relation'],
                        't': [''],
                        'x_type': x_type,
                        'y_type': y_type
                    }
                new_dialogue[1].append(new_relation)
            new_dialogues.append(new_dialogue)
        return new_dialogues

    def _dump_data(self, data, file_path):
        os.makedirs(Path(file_path).parents[0], exist_ok=True)
        with open(file_path, 'w', encoding='utf8') as file:
            json.dump(data, file)

    def _dump_relation_label_dict(self, data, output_path):
        # Flatten the data into a list of dictionaries
        flat_data = [item for sublist in data for item in sublist[1]]

        # Create a dataframe from the flattened data
        df = pd.DataFrame(flat_data)

        # Take the first element of each list in 'rid' and 'r' columns
        df['rid'] = df['rid'].apply(lambda x: x[0])
        df['r'] = df['r'].apply(lambda x: x[0])

        # Extract unique (rid, r) pairs, and convert the DataFrame to a dictionary
        label_dict = df[['rid', 'r']].drop_duplicates().set_index('rid').to_dict()['r']

        # Sort the dictionary by keys
        sorted_label_dict = {k: label_dict[k] for k in sorted(label_dict)}

        # Save the label dictionary to json file
        with open(output_path, 'w', encoding='utf8') as file:
            json.dump(sorted_label_dict, file)

        print(f"Label dictionary saved to {output_path}")

    def _overwrite_relations(self, data):
        for item in data:
            # item[1] corresponds to the list of relations
            for rel in item[1]:
                # Check if the relation type is 'no_relation'
                if rel['r'][0] == 'no_relation' or rel['r'][0] == 'unanswerable':
                    rel['r'] = ["no_relation"]
                    rel['rid'][0] = 0  # Set 'rid' to 0 for 'no_relation'
                # Check if the relation type is 'unanswerable'
                elif rel['r'][0] == 'inverse_relation':
                    rel['r'] = ["inverse_relation"]
                    rel['rid'] = [2]  # Set 'rid' to 1 for 'unanswerable'
                else:
                    rel['r'] = ["with_relation"]
                    rel['rid'] = [1]  # Set 'rid' to 2 for 'with_relation'
        return data

    def _merge_unanswerable_and_no_relation(self, data):
        for item in data:
            for rel in item[1]:
                # Check if the relation type is 'no_relation' or 'unanswerable'
                if rel['r'][0] == 'no_relation' or rel['r'][0] == 'unanswerable':
                    rel['r'] = ["no_relation_unanswerable" ]
                    rel['rid'] = [0]  # Set 'rid' to 0 for 'no_relation' and 'unanswerable'
                else:
                    rel['r'] = ["with_relation"]
                    rel['rid'] = [1]  # Set 'rid' to 1 for 'with_relation'
        return data

    def transform_to_binary(self,
                            input_folder=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-with-no-relation',
                            output_folder=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-binary'):
        
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"The folder '{input_folder}' does not exist. Please run method `add_no_relation_labels` first.")

        os.makedirs(output_folder, exist_ok=True)
        files = [Path(f) for f in glob.glob(str(input_folder / "*.json")) if 'relation_label_dict.json' not in str(f)]

        for file in files:
            with open(file, 'r', encoding='utf8') as json_file:
                data = json.load(json_file)

            # Merge 'unanswerable' and 'no_relation', and rename all other relations to 'with_relation'
            data = self._merge_unanswerable_and_no_relation(data)

            # Determine the set (train, dev, test) based on the filename
            set_type = file.stem.split('_')[-1]  # This assumes that the set type is always at the end of the file name

            # Write back to a new JSON file
            with open(output_folder / f"{set_type}.json", 'w', encoding='utf8') as json_file:
                json.dump(data, json_file)

        # Dump the new label dictionary
        self._dump_relation_label_dict(data, output_folder / 'relation_label_dict.json')

    def transform_to_ternary(self,
                             input_folder=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-with-no-relation',
                             output_folder=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-ternary'):

        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"The folder '{input_folder}' does not exist. Please run method `add_no_relation_labels` first.")

        os.makedirs(output_folder, exist_ok=True)
        files = [Path(f) for f in glob.glob(str(input_folder / "*.json")) if 'relation_label_dict.json' not in str(f)]

        for file in files:
            with open(file, 'r', encoding='utf8') as json_file:
                data = json.load(json_file)

            # Overwrite relations
            data = self._overwrite_relations(data)

            # Determine the set (train, dev, test) based on the filename
            set_type = file.stem.split('_')[-1]  # This assumes that the set type is always at the end of the file name

            # Write back to a new JSON file
            with open(output_folder / f"{set_type}.json", 'w', encoding='utf8') as json_file:
                json.dump(data, json_file)

        # Dump the new label dictionary
        self._dump_relation_label_dict(data, output_folder / 'relation_label_dict.json')

    def add_no_relation_labels(self,
                               output_folder=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-with-no-relation'):

        os.makedirs(output_folder, exist_ok=True)
        for filename in os.listdir(self.raw_data_folder):
            if 'relation_label_dict' in filename:
                continue
            if filename.endswith('.json'):
                input_file_path = os.path.join(self.raw_data_folder, filename)
                data = self._load_data(input_file_path)
                all_new_relations = []
                for dialogue in data:
                    relation_pairs = self._find_relation_pairs(dialogue)
                    all_possible_relations = self._find_all_relation_permutations(relation_pairs)
                    new_relations = self._exclude_existing_relations(dialogue, all_possible_relations)
                    all_new_relations.append(new_relations)

                new_data = self._create_new_dialogues_with_new_relations(data, all_new_relations)

                output_file_path = os.path.join(output_folder, filename)
                self._dump_data(new_data, output_file_path)

        # Dump the new label dictionary
        self._dump_relation_label_dict(new_data, output_folder / 'relation_label_dict.json')


class DialogREDatasetBalancer(DialogREDatasetTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _filter_dialogues(self, data):
        all_relations = set()
        for dialogue in data:
            all_relations.update(relation['r'][0] for relation in dialogue[1])

        filtered_data = [dialogue for dialogue in data if all_relations.issubset(relation['r'][0] for relation in dialogue[1])]
        
        print(f"Original dialogue count: {len(data)}, Filtered dialogue count: {len(filtered_data)}")
        print(f"Original relations count: {sum(len(dialogue[1]) for dialogue in data)}, Filtered relations count: {sum(len(dialogue[1]) for dialogue in filtered_data)}")

        return filtered_data

    def _resample_dialogue(self, dialogue, sampler):
        X = [[i] for i in range(len(dialogue[1]))]
        y = [relation['r'][0] for relation in dialogue[1]]

        X_res, _ = sampler.fit_resample(X, y)
        resampled_relations = [dialogue[1][i[0]] for i in X_res]

        return [dialogue[0], resampled_relations]

    def _resample(self, data, sampler):
        resampled_data = [self._resample_dialogue(dialogue, sampler) for dialogue in data]

        return resampled_data
    
    def _copy_other_files(self, input_folder, output_folder, ignore_files=None):
        """
        Copy all files from the input folder to the output folder,
        excluding the ones specified in the ignore_files list.
        """
        for filename in os.listdir(input_folder):
            if filename in ignore_files:
                continue

            shutil.copy(os.path.join(input_folder, filename),
                        os.path.join(output_folder, filename)) 

    def undersample(self, train_file, output_folder):
        data = self._load_data(train_file)
        filtered_data = self._filter_dialogues(data)
        filtered_data = data
        
        undersampler = RandomUnderSampler(random_state=42)
        resampled_data = self._resample(filtered_data, undersampler)

        output_file_path = os.path.join(output_folder, train_file.name)
        self._dump_data(resampled_data, output_file_path)
        self._copy_other_files(train_file.parents[0], output_folder, ignore_files=['train.json'])


    def oversample(self, train_file, output_folder):
        data = self._load_data(train_file)
        filtered_data = self._filter_dialogues(data)

        oversampler = RandomOverSampler(random_state=42)
        resampled_data = self._resample(filtered_data, oversampler)

        output_file_path = os.path.join(output_folder, train_file.name)
        self._dump_data(resampled_data, output_file_path)
        self._copy_other_files(train_file.parents[0], output_folder, ignore_files=['train.json'])


class DialogRERelationEnricher:
    def __init__(self):
        self.enricher = DialogueEnricher()

    def process_dialogues(self, input_dir: str, output_dir: str):
        self._check_and_create_output_dir(output_dir)
        for filename in os.listdir(input_dir):
            print("Processing file: {}".format(filename))
            self._process_files_in_dir(filename, input_dir, output_dir)

    def _check_and_create_output_dir(self, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _process_files_in_dir(self, filename: str, input_dir: str, output_dir: str):
        if filename.endswith(".json"):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)

            if filename == "relation_label_dict.json":
                shutil.copyfile(input_file, output_file)
            else:
                self._process_json_file(input_file, output_file)

    def _process_json_file(self, input_file: str, output_file: str):
        with open(input_file, 'r', encoding='utf8') as f:
            dialogues_relations = json.load(f)
        processed_dialogues = self.enricher.enrich(dialogues_relations)
        self._save_processed_dialogues(processed_dialogues, output_file)

    def _save_processed_dialogues(self, processed_dialogues, output_file):
        with open(output_file, 'w', encoding='utf8') as f:
            json.dump(processed_dialogues, f)
            





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
    
    ### FIX METRICS COHERENCE ##
    # Drop rows where 'r' is None or NaN
    x_df.dropna(subset=['r'], inplace=True)
    x_df['dialog'] = x_df['dialog'].astype(str)
    # Now backtrack to create the other DataFrames
    final_df = x_df.groupby(['dialog', 'dataset_name']).first().reset_index()
    ### FIX METRICS COHERENCE ##

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

def undersample_dialogre(input_path, output_path, binary=False):
    if not binary:
        # List all files in the directory and filter out irrelevant ones
        files = os.listdir(input_path)
        files = [f for f in files if f != "relation_label_dict.json" and f.endswith('.json')]

        # Initialize an empty list to store DataFrames
        data_frames = []

        # Read and process each file
        for file in files:
            with open(os.path.join(input_path, file), 'r') as f:
                data = json.load(f)

            # Convert each file's data into a DataFrame
            df = pd.DataFrame(data, columns=['dialog', 'relation_info'])
            df['dataset_name'] = file.replace('.json', '')
            data_frames.append(df)

        # Combine all dataframes
        final_df = pd.concat(data_frames, ignore_index=True)

        # Explode DataFrame to expand 'relation_info'
        x_df = final_df.explode('relation_info')

        # Remove rows where 'relation_info' is NaN
        mask = x_df['relation_info'].notna()
        x_df = x_df[mask]

        # Create new column 'r' based on 'rid' and 'r' values in 'relation_info'
        x_df['r'] = x_df['relation_info'].dropna().apply(lambda x: f"{x['rid'][0]:02}__{x['r'][0]}")

        # Create pivot table
        pivot_df = pd.pivot_table(x_df, values='relation_info', index=['r'], columns=['dataset_name'], aggfunc='count', fill_value=0)
        pivot_df.columns = [f'count_{col}' for col in pivot_df.columns]
        pivot_df['total'] = pivot_df.sum(axis=1)
        pivot_df['proportion'] = (pivot_df['total'] / pivot_df['total'].sum() * 100).apply(lambda x: f"{x:.1f}%")
        pivot_df.reset_index(inplace=True)
        pivot_df[['rid', 'r']] = pivot_df['r'].str.split('__', expand=True)
        pivot_df = pivot_df.sort_values('total', ascending=False)
        id_to_relation = pivot_df.set_index('rid')['r'].to_dict()
        relation_to_id = {v:k for k,v in id_to_relation.items()}
        null_relation_id = relation_to_id.get('null_relation', relation_to_id.get('no_relation'))
        
        # Filter DataFrames based on 'rid' (dumps no_relation and inverse_relation)
        filtered_df = pivot_df[pivot_df['rid'].astype(int) != int(null_relation_id)]

        top_item = filtered_df.iloc[0]
        top_item_counts = {k.split('_')[1]: v for k, v in top_item.to_dict().items() if 'count' in k}

        # Perform random sampling and combine DataFrames
        exploded_df = final_df.explode('relation_info')
        exploded_df['rid'] = exploded_df['relation_info'].apply(lambda x: x['rid'][0] if isinstance(x, dict) else 39)
        filtered_df = exploded_df[exploded_df['rid'] != int(null_relation_id)]
        no_relation_df = exploded_df[exploded_df['rid'] == int(null_relation_id)]

        for dataset_name, count in top_item_counts.items():
            subset = no_relation_df[no_relation_df['dataset_name'] == dataset_name]
            random_subset = subset.sample(n=count, random_state=1)
            filtered_df = pd.concat([filtered_df, random_subset])

        # Perform aggregation
        unexploded_df = filtered_df.groupby(filtered_df.index).agg({
            'dialog': 'first',
            'relation_info': lambda x: x.dropna().tolist(),
            'dataset_name': 'first'
        })
        unexploded_df.reset_index(drop=True, inplace=True)

        # Ensure output directory exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Save processed DataFrames to JSON files
        for dataset_name in unexploded_df['dataset_name'].unique():
            subset_df = unexploded_df[unexploded_df['dataset_name'] == dataset_name]
            subset_df = subset_df.drop(columns=['dataset_name'])
            output_data = []
            for _, row in subset_df.iterrows():
                dialog = row['dialog']
                relation_info = row['relation_info']
                output_data.append([dialog, relation_info])
            output_file_path = os.path.join(output_path, f"{dataset_name}.json")
            with open(output_file_path, 'w') as f:
                json.dump(output_data, f)
        
    else:
        files = os.listdir(input_path)
        data_frames = []

        # Filter relevant json files
        files = [f for f in files if f != "relation_label_dict.json" and f.endswith('.json')]

        # Read and load json files into dataframes
        for file in files:
            with open(os.path.join(input_path, file), 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data, columns=['dialog', 'relation_info'])
            df['dataset_name'] = file.replace('.json', '')
            data_frames.append(df)

        # Merge all dataframes
        final_df = pd.concat(data_frames, ignore_index=True)

        # Explode relation_info and filter out nulls
        x_df = final_df.explode('relation_info')
        mask = x_df['relation_info'].notna()
        x_df = x_df[mask]
        x_df['r'] = x_df['relation_info'].dropna().apply(lambda x: f"{x['rid'][0]:02}__{x['r'][0]}")

        # Creating a pivot table to summarize relation information
        pivot_df = pd.pivot_table(x_df, values='relation_info', index=['r'], columns=['dataset_name'], aggfunc='count', fill_value=0)
        pivot_df.columns = [f'count_{col}' for col in pivot_df.columns]
        pivot_df['total'] = pivot_df.sum(axis=1)
        pivot_df['proportion'] = (pivot_df['total'] / pivot_df['total'].sum() * 100).apply(lambda x: f"{x:.1f}%")
        pivot_df.reset_index(inplace=True)
        pivot_df[['rid', 'r']] = pivot_df['r'].str.split('__', expand=True)
        pivot_df = pivot_df.sort_values('total', ascending=False)
        id_to_relation = pivot_df.set_index('rid')['r'].to_dict()
        relation_to_id = {v:k for k,v in id_to_relation.items()}
        null_relation_id = relation_to_id.get('null_relation', relation_to_id.get('no_relation'))

        # Filter rows where 'rid' == 2
        filtered_df = pivot_df[pivot_df['rid'].astype(int) != int(null_relation_id)] # with_relation
        top_item = filtered_df.iloc[0]
        top_item_counts = {k.split('_')[1]: v for k,v in top_item.to_dict().items() if 'count' in k} # get counts

        import random

        # Explode and filter rows
        exploded_df = final_df.explode('relation_info')
        exploded_df['rid'] = exploded_df['relation_info'].apply(lambda x: x['rid'][0] if isinstance(x, dict) else 39)
        filtered_df = exploded_df[exploded_df['rid'] != int(null_relation_id)] # with_relation
        no_relation_df = exploded_df[exploded_df['rid'] == int(null_relation_id)] # no_relation

        # Balance data based on top_item
        for dataset_name, count in top_item_counts.items():
            subset = no_relation_df[no_relation_df['dataset_name'] == dataset_name]
            random_subset = subset.sample(n=count, random_state=1)
            filtered_df = pd.concat([filtered_df, random_subset])

        # Aggregate and finalize the dataframe
        unexploded_df = filtered_df.groupby(filtered_df.index).agg({
            'dialog': 'first',
            'relation_info': lambda x: x.dropna().tolist(),
            'dataset_name': 'first'
        })
        unexploded_df.reset_index(drop=True, inplace=True)

        # Create output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Save processed data into JSON files
        for dataset_name in unexploded_df['dataset_name'].unique():
            subset_df = unexploded_df[unexploded_df['dataset_name'] == dataset_name]
            subset_df = subset_df.drop(columns=['dataset_name'])
            output_data = []
            for _, row in subset_df.iterrows():
                dialog = row['dialog']
                relation_info = row['relation_info']
                output_data.append([dialog, relation_info])
            output_file_path = os.path.join(output_path, f"{dataset_name}.json")
            with open(output_file_path, 'w') as f:
                json.dump(output_data, f)
        
    dump_readme(output_path)