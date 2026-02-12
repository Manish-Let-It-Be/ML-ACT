import pandas as pd
import numpy as np
import pickle
import os
from config import MODELS_DIR


def save_model(model, name):
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    path = os.path.join(MODELS_DIR, f"{safe_name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_dataset(filepath):
    df = pd.read_csv(filepath)
    return df


def get_dataset_info(df, name="Dataset"):
    info = {
        "name": name,
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.value_counts().to_dict(),
        "missing_values": df.isnull().sum().sum(),
        "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
    }
    return info


def format_dataset_info(info):
    lines = [
        f"Dataset: {info['name']}",
        f"Shape: {info['shape'][0]} rows x {info['shape'][1]} columns",
        f"Numeric columns: {info['numeric_columns']}",
        f"Missing values: {info['missing_values']}",
    ]
    return "\n".join(lines)
