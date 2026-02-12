import numpy as np
import pandas as pd


def remove_outliers_zscore(df, threshold=3.0):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
    mask = (z_scores < threshold).all(axis=1)
    return df[mask].reset_index(drop=True)
