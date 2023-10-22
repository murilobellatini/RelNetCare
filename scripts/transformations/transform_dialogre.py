from src.processing.dialogre_processing import DialogREDatasetTransformer, dump_readme
from src.paths import LOCAL_PROCESSED_DATA_PATH


if __name__ == "__main__":
    # Create an instance of the DialogREDatasetResampler
    resampler = DialogREDatasetTransformer(raw_data_folder='/home/murilo/RelNetCare/data/processed/dialog-re-36cls')
    
    no_rel_folder=  LOCAL_PROCESSED_DATA_PATH / 'dialog-re-38cls-with-no-and-inverse-relation'
    binary_folder=  LOCAL_PROCESSED_DATA_PATH / 'dialog-re-2cls-binary'
    # Add 'no_relation' to dialogues where there are no relation mentions
    # resampler.add_no_relation_labels(output_folder=no_rel_folder)
    # dump_readme(no_rel_folder)

    # Convert the dataset into a binary classification problem by merging 
    # 'unanswerable' to 'no_relation' and 'inverse_relation' to 'with_relation'
    # output classes: 'no_relation' and 'with_relation'
    resampler.transform_to_binary(input_folder=no_rel_folder,
                                  output_folder=binary_folder)
    dump_readme(binary_folder)


    # # Convert the dataset into a ternary classification problem by merging 'unanswerable' and 'no_relation', 
    # # keeping 'inverse_relation' class and overwrite all other relations with 'with_relation'
    # # output classes: 'no_relation', 'with_relation' and 'inverse_relation'
    # resampler.transform_to_ternary(input_folder=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-with-no-relation',
                                #    output_folder=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-ternary')
