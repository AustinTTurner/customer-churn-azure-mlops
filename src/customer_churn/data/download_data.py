"""Download the Telco Customer Churn dataset"""

from pathlib import Path

import openml

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_FILE = RAW_DATA_DIR / "telco_customer_churn.csv"

OPENML_DATASET_ID = 42178

def download_dataset() -> Path:
    """Download the Telco Customer Churn dataset from OpenML"""

    RAW_DATA_DIR.mkdir(parents = True, exist_ok = True)

    print("Downloading Telco Customer Churn dataset...")

    dataset = openml.datasets.get_dataset(OPENML_DATASET_ID)

    features, target, _, _ = dataset.get_data(
        target = dataset.default_target_attribute,
        include_ignore_attribute = True,
    )

    dataframe = features.copy()
    dataframe["Churn"] = target

    if "customerID" not in dataframe.columns:
        raise RuntimeError(
            "Expected customerID column was not found in the downloaded dataset."
        )

    dataframe.to_csv(RAW_DATA_FILE, index = False)

    print(f"Dataset saved to: {RAW_DATA_FILE}")
    print(F"Rows: {dataframe.shape[0]}")
    print(f"Colums: {dataframe.shape[1]}")

    return RAW_DATA_FILE

if __name__ == "__main__":
    download_dataset()