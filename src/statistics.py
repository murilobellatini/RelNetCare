import pandas as pd


def get_counts_and_percentages(sr:pd.Series):
    sr_counts = sr.value_counts()
    sr_pct = sr.value_counts(normalize=True).mul(100).round(1)
    return pd.concat([sr_counts, sr_pct], axis=1, keys=['Counts', '%'])
