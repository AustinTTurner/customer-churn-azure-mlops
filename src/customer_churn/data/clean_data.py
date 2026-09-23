"""Clean and preprocess the Telco Customer Churn dataset"""

from pathlib import Path

import pandas as pd

from customer_churn.data.validate_data import (
    load_raw_data,
    validate_dataframe,
)

PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_DATA_FILE = (
    PROCESSED_DATA_DIR / "telco_customer_churn_clean.csv"
)

def clean_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean raw customer churn data"""

    cleaned = dataframe.copy()

    # Convert TotalCharges from text/object to numeric
    # Blank or invalid values become NaN
    cleaned["TotalCharges"] = pd.to_numeric(
        cleaned["TotalCharges"],
        errors = "coerce",
    )

    # New customers with tenure == 0 have not accumulated charges yet
    zero_tenure_mask = (
        cleaned["TotalCharges"].isna()
        & cleaned["tenure"].eq(0)
    )

    cleaned.loc[zero_tenure_mask, "TotalCharges"] = 0.0

    return cleaned

def validate_cleaned_data(dataframe: pd.DataFrame) -> None:
    """Validate expectations after cleaning"""

    missing_total_charges = dataframe["TotalCharges"].isna().sum()

    if missing_total_charges > 0:
        raise ValueError(
            "TotalCharges still contains "
            f"{missing_total_charges} missing values after cleaning."
        )

    if (dataframe["TotalCharges"] < 0).any():
        raise ValueError(
            "TotalCharges contains negative values."
        )

def save_processed_data(
        dataframe: pd.DataFrame,
        path: Path = PROCESSED_DATA_FILE,
) -> Path:
    """Save cleaned data to the processed directory"""

    PROCESSED_DATA_DIR.mkdir(
        parents = True,
        exist_ok = True,
    )

    dataframe.to_csv(path, index = False)

    return path

def main() -> None:
    """Run the raw-to-clean data pipeline"""

    dataframe = load_raw_data()

    print("Validating raw dataset...")
    validate_dataframe(dataframe)

    print("Cleaning dataset...")
    cleaned = clean_dataframe(dataframe)

    validate_cleaned_data(cleaned)

    output_path = save_processed_data(cleaned)

    print("Data cleaning completed.")
    print(f"Rows: {cleaned.shape[0]}")
    print(f"Columns: {cleaned.shape[1]}")
    print(
        "Missing TotalCharges values:",
        cleaned["TotalCharges"].isna().sum(),
    )
    print(
        "TotalCharges dtype:",
        cleaned["TotalCharges"].dtype,
    )
    print(f"Saved processed dataset to: {output_path}")


if __name__ == "__main__":
    main()