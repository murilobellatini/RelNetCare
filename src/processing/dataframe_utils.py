import pandas as pd


def get_counts_and_percentages(df: pd.DataFrame, cols: list):
    sr_counts = df.groupby(cols).size()
    total_counts = sr_counts.sum()
    sr_pct = (sr_counts / total_counts).mul(100).round(1)
    return pd.concat([sr_counts, sr_pct], axis=1, keys=['Counts', '%']).sort_values('Counts', ascending=False)
