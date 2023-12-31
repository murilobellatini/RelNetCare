"""
All relevant paths stored in constant variables
"""

import pathlib as pl

# local paths
root_path = pl.Path(__file__).parents[1]
LOCAL_DATA_PATH           = root_path / 'data'
LOCAL_RAW_DATA_PATH       = LOCAL_DATA_PATH / 'raw'
LOCAL_INTERIM_DATA_PATH   = LOCAL_DATA_PATH / 'interim'
LOCAL_PROCESSED_DATA_PATH = LOCAL_DATA_PATH / 'processed'
LOCAL_MODELS_PATH         = pl.Path('/mnt/vdb1/murilo/models')
LOCAL_CREDENTIALS_PATH    = root_path / 'credentials'
LOCAL_LOGGING_PATH        = root_path / 'logging/log.log'
LOCAL_CONFIG_PATH         = root_path / 'src/config'

NOTEBOOKS_PATH = root_path / 'notebooks'
BASELINES_PATH = root_path / 'Baselines'

# global paths
GLOBAL_DATA_PATH          = pl.Path('data')


