from src.processing import DialogREDatasetTransformer
from src.paths import LOCAL_PROCESSED_DATA_PATH


if __name__ == "__main__":
    # Create an instance of the DialogREDatasetResampler
    resampler = DialogREDatasetTransformer()

    # Add 'no_relation' to dialogues where there are no relation mentions
    resampler.add_no_relation_labels(output_folder=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-with-no-relation')

    # Convert the dataset into a binary classification problem by merging 'unanswerable' and 'no_relation'
    resampler.transform_to_binary(input_folder=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-with-no-relation',
                                  output_folder=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-binary')

    # Convert the dataset into a ternary classification problem by merging 'unanswerable' and 'no_relation', 
    # and overwrite all other relations with 'with_relation'
    resampler.transform_to_ternary(input_folder=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-with-no-relation',
                                   output_folder=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-ternary')
