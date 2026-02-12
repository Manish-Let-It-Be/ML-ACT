import os
import pandas as pd
import zipfile
import glob


def download_kaggle_dataset(dataset_name, download_dir="kaggle_datasets"):
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        os.makedirs(download_dir, exist_ok=True)
        api.dataset_download_files(dataset_name, path=download_dir, unzip=True)

        csv_files = glob.glob(os.path.join(download_dir, "*.csv"))
        if not csv_files:
            for root, dirs, files in os.walk(download_dir):
                for f in files:
                    if f.endswith(".csv"):
                        csv_files.append(os.path.join(root, f))

        if csv_files:
            largest = max(csv_files, key=os.path.getsize)
            df = pd.read_csv(largest)
            return df, os.path.basename(largest), None
        return None, None, "No CSV files found in the downloaded dataset."
    except Exception as e:
        return None, None, str(e)


def detect_target_column(df):
    common_targets = ["target", "label", "class", "y", "output", "outcome", "survived", "price"]
    for col in common_targets:
        if col in df.columns.str.lower():
            idx = df.columns.str.lower().tolist().index(col)
            return df.columns[idx]
    return df.columns[-1]
