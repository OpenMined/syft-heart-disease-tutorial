import requests
import os

import pandas as pd
import numpy as np

from typing import Optional
from zipfile import ZipFile
from io import BytesIO


# ==============

DATA_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal", "ca"]

FEATURE_RANGES = {
    "age": range(1, 100),
    "sex": range(2),
    "cp": range(1, 5),
    "trestbps": range(60, 210),
    "chol": range(50, 600),
    "fbs": range(2),
    "restecg": range(3),
    "thalach": range(60, 210),
    "exang": range(2),
    "oldpeak": np.arange(-2, 5, step=0.1),
    "slope": range(1, 4),
    "ca": range(0, 4),
    "thal": (3, 6, 7),
    "num": range(5),
}

CLEVELAND = "Cleveland Clinic"
HUNGARY = "Hungarian Inst. of Cardiology"
SWITZERLAND = "Univ. Hospitals Zurich and Basel"
LONG_BEACH = "V.A. Medical Center"

DATASETS = {
    CLEVELAND: "processed.cleveland.data",
    HUNGARY: "processed.hungarian.data",
    SWITZERLAND: "processed.switzerland.data",
    LONG_BEACH: "processed.va.data",
}

NAMES = tuple(DATASETS.keys())

# ==============


def download_data(root: str = "./tmp/data", data_url: str = DATA_URL) -> bool:
    r = requests.get(data_url)
    if r.status_code != 200:
        print("Data URL invalid, or incorrect. Please check!")
        return False
    archive = ZipFile(BytesIO(r.content))
    archive.extractall(root)
    print(f"Data successfully downloaded in {root}")
    return True


def load_data(
    name: str, root: str = "./tmp/data", data_url: str = DATA_URL
) -> Optional[pd.DataFrame]:
    """Load the Heart-disease dataset from the selected hospital - identified
    by the input `name`. The full data package will be downloaded first,
    if it cannot be found in the selected `root` folder.
    """
    data_key = name if name in DATASETS else None
    if data_key is None:
        print(f"Name of Datasite {name} is invalid, or incorrect. Please check!")
        return None

    file_path = os.path.join(root, DATASETS[data_key])
    if not os.path.exists(file_path):
        downloaded = download_data(root=root, data_url=data_url)
        if not downloaded:
            return None

    df = pd.read_csv(file_path, header=None, index_col=False, names=FEATURES + ["num"])
    # convert all data as numeric format (forcing downcasting to int when possible)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], downcast="integer", errors="coerce")

    return df


def generate_mock(data: pd.DataFrame, seed: int = 12345) -> pd.DataFrame:
    """Generates random mock heart-disease data, given an input seed.

    The number of samples in each dataset will be completely random.
    However, for each colum (i.e. feature):
        - data values will be mapped to the same original domain;
        - the rate of missing values (if any) will be kept the same, proportionally.
    """

    np.random.seed(seed=seed)
    mock_n_samples = np.random.choice(np.arange(50, 300))
    data_info = {}
    for column, feature_range in FEATURE_RANGES.items():
        rnd_values = np.random.choice(feature_range, size=mock_n_samples)
        true_na_rate = data[column].isna().sum()
        if true_na_rate > 0:
            true_n_samples = data.shape[0]
            mock_na_rate = int(
                np.rint((true_na_rate * mock_n_samples) / true_n_samples)
            )
            rnd_indices = np.random.choice(np.arange(mock_n_samples), size=mock_na_rate)
            rnd_values = rnd_values.astype(np.float32)
            rnd_values[rnd_indices] = np.nan
        data_info[column] = rnd_values
    return pd.DataFrame(data_info)
