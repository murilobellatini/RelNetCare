# Import required modules
import os
import json
import pandas as pd
import random
from src.processing.dialogre_processing import undersample_dialogre

# # Set paths for data and output
# input_path = "/home/murilo/RelNetCare/data/processed/dialog-re-38cls-with-no-and-inverse-relation"
# output_path = "/home/murilo/RelNetCare/data/processed/dialog-re-38cls-with-no-and-inverse-relation-undersampled"

# undersample_dialogre(input_path, output_path)




# Set paths for data and output
input_path = "/home/murilo/RelNetCare/data/processed/dialog-re-12cls"
output_path = "/home/murilo/RelNetCare/data/processed/dialog-re-12cls-with-no-relation-undersampled"

undersample_dialogre(input_path, output_path, binary=False)