"""Download and prepare the Diabetes 130‑US hospitals dataset.

This script downloads the CSV from the UCI Machine Learning Repository, performs a few basic
cleaning steps, and writes a processed version to `data/diabetic_data.csv`. You can run this
script from the project root via:

```
python src/data.py
```
"""

import os
import pathlib
import zipfile
from urllib import request

import pandas as pd

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"
)
DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"
RAW_FILE = DATA_DIR / "dataset_diabetes.zip"
CSV_FILE = DATA_DIR / "diabetic_data.csv"


def download_data() -> None:
    """Download the compressed dataset if it doesn't already exist."""
    DATA_DIR.mkdir(exist_ok=True)
    if not RAW_FILE.exists():
        print(f"Downloading dataset to {RAW_FILE}...")
        request.urlretrieve(DATA_URL, RAW_FILE)
        print("Download complete.")
    else:
        print("Raw dataset already downloaded.")


def extract_csv() -> None:
    """Extract the CSV from the downloaded zip file."""
    if not CSV_FILE.exists():
        print(f"Extracting CSV to {CSV_FILE}...")
        with zipfile.ZipFile(RAW_FILE, "r") as zf:
            for member in zf.namelist():
                if member.lower().endswith("diabetic_data.csv"):
                    with zf.open(member) as src, open(CSV_FILE, "wb") as dst:
                        dst.write(src.read())
                    break
        print("Extraction complete.")
    else:
        print("CSV already extracted.")


def prepare_data() -> pd.DataFrame:
    """Load the CSV into a pandas DataFrame and perform basic cleaning.

    Returns the cleaned DataFrame.
    """
    print("Loading CSV...")
    df = pd.read_csv(CSV_FILE)
    # Drop columns with too many missing values or identifiers
    drop_cols = ["encounter_id", "patient_nbr"]
    df = df.drop(columns=drop_cols)
    # Replace '?' placeholders with NaN
    df = df.replace("?", pd.NA)
    # Convert target 'readmitted' into binary: 1 if <30, 0 otherwise
    df["readmitted_flag"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)
    return df


def main() -> None:
    download_data()
    extract_csv()
    df = prepare_data()
    # Save processed data for later use
    output_path = DATA_DIR / "processed_diabetic_data.csv"
    print(f"Saving processed data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
