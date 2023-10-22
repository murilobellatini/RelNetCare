import os
import json
import pandas as pd
from pandas import json_normalize

# Initialize an empty list to collect the data
data_list = []

# Starting directory
root_dir = "/home/murilo/RelNetCare/data/reports"

# Walk through directory recursively
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file == "class_metrics.json":
            # Extract dataset_name and model_name
            path_parts = root.split("/")
            root_parts = root_dir.split("/")
            dataset_name_position = len(root_parts)
            dataset_name = path_parts[dataset_name_position]
            model_name = path_parts[-1]

            
            # Read json file
            with open(os.path.join(root, file), "r") as f:
                metrics = json.load(f)
            
            # Add dataset_name and model_name to metrics
            metrics['dataset_name'] = dataset_name
            metrics['model_name'] = model_name
            
            # Append to data list
            data_list.append(metrics)

# Convert list of dicts to DataFrame
df = pd.DataFrame(data_list)



# Initialize an empty list to collect the normalized data
normalized_data_list = []

# Iterate through each row in the DataFrame
for idx, row in df.iterrows():
    # Extract metrics, dataset_name, and model_name
    metrics = row.to_dict()
    dataset_name = metrics.pop('dataset_name')
    model_name = metrics.pop('model_name')
    
    # Normalize the nested dictionaries
    normalized_row = json_normalize(metrics)
    
    # Add back the dataset_name and model_name
    normalized_row['dataset_name'] = dataset_name
    normalized_row['model_name'] = model_name
    
    # Append the normalized row to the list
    normalized_data_list.append(normalized_row)

# Concatenate all the normalized rows into a new DataFrame
normalized_df = pd.concat(normalized_data_list, ignore_index=True)

# Now 'normalized_df' should be denormalized
normalized_df

columns_order = [
    'exp_group',
    'model_name',
    'dataset_name',
]

# List of class names you want to include
class_names = [
    'null_relation',
    'place_of_residence',
    'residents_of_place',
    'siblings',
    'spouse',
    'other_family',
    'pet',
    'children',
    'parents',
    'acquaintance',
    'visited_place',
    'visitors_of_place',
]

# Extract the column names from normalized_df
normalized_columns = normalized_df.columns

# Create a temporary variable for missing columns
missing_columns = []

# Add any missing columns from normalized_df to class_names
for col in normalized_columns:
    if col.startswith('per_class.') and col.split('.')[1] not in class_names:
        missing_columns.append(col.split('.')[1])
        class_names.append(col.split('.')[1])

# Check if there are missing columns and print a warning
if missing_columns:
    print(f"Warning: The following columns were missing and have been added to class_names: {missing_columns}")

# Add precision, recall, and f1 for each class
columns_order.append(f'micro_avg.precision')
columns_order.append(f'micro_avg.recall')
columns_order.append(f'micro_avg.f1')
columns_order.append(f'macro_avg.precision')
columns_order.append(f'macro_avg.recall')
columns_order.append(f'macro_avg.f1')


for class_name in class_names:
    columns_order.append(f'per_class.{class_name}.precision')
    columns_order.append(f'per_class.{class_name}.recall')
    columns_order.append(f'per_class.{class_name}.f1')

# Now columns_order contains the desired column order
output_path = f'{root_dir}/analytics.csv'
print("output_path=", output_path)
normalized_df[columns_order].to_csv(output_path)

