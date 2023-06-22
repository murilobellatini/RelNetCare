from src.processing import DialogREDatasetFixer
from src.paths import LOCAL_PROCESSED_DATA_PATH

if __name__ == "__main__":
    fixer = DialogREDatasetFixer()
    fixer.process()
