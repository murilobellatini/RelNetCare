import pandas as pd


def get_pct_value_counts(sr:pd.Series):
    sr_counts = sr.value_counts()
    sr_pct = sr.value_counts(normalize=True)
    return pd.concat([sr_counts, sr_pct], axis=1)