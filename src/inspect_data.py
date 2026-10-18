from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/processed.cleveland.data")


COLUMNS = [
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
    "num",
]


def load_data():

    df = pd.read_csv(
        DATA_PATH,
        header=None,
        names=COLUMNS,
        na_values="?",
    )

    df["target"] = (df["num"] > 0).astype(int)

    return df


if __name__ == "__main__":
    df = load_data()

    print("Shape:", df.shape)

    print("\nFirst five rows:")

    print(df.head())

    print("\nMissing values:")

    print(df.isna().sum())

    print("\nOriginal target distribution:")

    print(df["num"].value_counts().sort_index())

    print("\nBinary target distribution:")

    print(df["target"].value_counts().sort_index())
