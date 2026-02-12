import pandas as pd
import numpy as np


def handle_missing_values(df, method="drop"):
    if method == "drop":
        return df.dropna()
    elif method == "mean":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        return df
    elif method == "median":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        return df
    return df
