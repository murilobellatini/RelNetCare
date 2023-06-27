from src.processing import DialogREDatasetResampler
from src.paths import LOCAL_PROCESSED_DATA_PATH

if __name__ == "__main__":
    resammpler = DialogREDatasetResampler()
    resammpler.add_no_relation(output_folder=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-with-no-relation')
    resammpler.make_binary(input_folder=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-with-no-relation',
                       output_folder=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-binary')
    resammpler.make_ternary(input_folder=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-with-no-relation',
                          output_folder=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-ternary')
