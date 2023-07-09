import os
import json
import copy
import glob
import nltk
import spacy
import shutil
import itertools
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


from src.processing.text_preprocessing import TextPositionTracker, EntityExtractor, SpacyFeatureExtractor
from src.processing.distance_computation import TurnDistanceCalculator, DistanceComputer
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
            with open(file_name, 'r') as file:
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
        with open(output_path, 'w') as file:
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
            with open(file, 'r') as json_file:
                data = json.load(json_file)

            # Merge 'unanswerable' and 'no_relation', and rename all other relations to 'with_relation'
            data = self._merge_unanswerable_and_no_relation(data)

            # Determine the set (train, dev, test) based on the filename
            set_type = file.stem.split('_')[-1]  # This assumes that the set type is always at the end of the file name

            # Write back to a new JSON file
            with open(output_folder / f"{set_type}.json", 'w') as json_file:
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
            with open(file, 'r') as json_file:
                data = json.load(json_file)

            # Overwrite relations
            data = self._overwrite_relations(data)

            # Determine the set (train, dev, test) based on the filename
            set_type = file.stem.split('_')[-1]  # This assumes that the set type is always at the end of the file name

            # Write back to a new JSON file
            with open(output_folder / f"{set_type}.json", 'w') as json_file:
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
        self.nlp = spacy.load("en_core_web_sm")
        nltk.download('punkt')
        self.txt_pos_tracker = TextPositionTracker(self.nlp)
        self.entity_extractor = EntityExtractor()
        self.feature_extractor = SpacyFeatureExtractor(self.nlp)
        self.turn_distance_calculator = TurnDistanceCalculator()
        self.distance_computer = DistanceComputer(self.txt_pos_tracker, self.entity_extractor, self.feature_extractor, self.turn_distance_calculator)

    def process_dialogues(self, input_dir: str, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for filename in os.listdir(input_dir):
            print("Processing file: {}".format(filename))
            if filename.endswith(".json"):
                input_file = os.path.join(input_dir, filename)
                output_file = os.path.join(output_dir, filename)

                if filename == "relation_label_dict.json":
                    shutil.copyfile(input_file, output_file)
                else:
                    with open(input_file, 'r', encoding='utf8') as f:
                        dialogues_relations = json.load(f)

                    processed_dialogues = []
                    for dialogue, relations in tqdm(dialogues_relations):
                        relations_with_distances = self.distance_computer.compute_distances(dialogue, relations)
                        processed_dialogues.append((dialogue, relations_with_distances))

                    # Assert that 'data' and 'new_data' have the same length
                    assert len(dialogues_relations) == len(processed_dialogues), "Data and new data have different lengths"

                    # Assert that 'data' and 'new_data' have the same relation count for each dialogue
                    for original_dialogue, new_dialogue in zip(dialogues_relations, processed_dialogues):
                        assert len(original_dialogue[1]) == len(new_dialogue[1]), "Original and new dialogues have different relation counts"

                    # Filter relations to only include those with a 'min_words_distance' key
                    for i, (dialogue, relations) in enumerate(processed_dialogues):
                        processed_dialogues[i] = (dialogue, [r for r in relations if 'min_words_distance' in r])

                    # Filter dialogues to only include those with at least one relation containing a 'min_words_distance' key
                    processed_dialogues = [(dialogue, relations) for (dialogue, relations) in processed_dialogues if any('min_words_distance' in r for r in relations)]

                    # Check that every new relation contains a 'min_words_distance' key
                    for _, relations in processed_dialogues:
                        for r in relations:
                            assert 'min_words_distance' in r, "Item in new relation does not contain 'min_words_distance' key"
                            
                    with open(output_file, 'w', encoding='utf8') as f:
                        json.dump(processed_dialogues, f)
