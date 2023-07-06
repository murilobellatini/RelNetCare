from src.processing.etl import DialogREDatasetBalancer
from src.paths import LOCAL_PROCESSED_DATA_PATH


if __name__ == "__main__":
    
    data_dir_stem = 'dialog-re-binary-enriched'
    sampler = DialogREDatasetBalancer()
    sampler.undersample(train_file=LOCAL_PROCESSED_DATA_PATH / f'{data_dir_stem}/train.json',
                        output_folder=LOCAL_PROCESSED_DATA_PATH / f'{data_dir_stem}-undersampled')
    sampler.oversample(train_file=LOCAL_PROCESSED_DATA_PATH / f'{data_dir_stem}/train.json',
                        output_folder=LOCAL_PROCESSED_DATA_PATH / f'{data_dir_stem}-oversampled')
