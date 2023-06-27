from src.processing import DialogREDatasetResampler
from src.paths import LOCAL_PROCESSED_DATA_PATH

if __name__ == "__main__":
    fixer = DialogREDatasetResampler()
    fixer.add_no_relation()
