from src.processing import DialogREDatasetBalancer
from src.paths import LOCAL_PROCESSED_DATA_PATH


if __name__ == "__main__":
    sampler = DialogREDatasetBalancer(raw_data_folder='path_to_raw_data')
    sampler.undersample(train_file=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-ternary/train.json',
                        output_folder=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-ternary-undersampled')
    sampler.oversample(train_file=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-ternary/train.json',
                        output_folder=LOCAL_PROCESSED_DATA_PATH / 'dialog-re-ternary-oversampled')
