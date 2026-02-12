from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np


def normalize_data(X, method="standard"):
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        return X, None

    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
