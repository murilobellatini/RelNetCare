from src.processing import DialogREDatasetResampler
from src.paths import LOCAL_PROCESSED_DATA_PATH

if __name__ == "__main__":
    fixer = DialogREDatasetResampler()
    fixer.add_no_relation(output_folder=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-with-no-relation')
    fixer.make_ternary(input_folder=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-with-no-relation',
                       output_folder=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-ternary')
