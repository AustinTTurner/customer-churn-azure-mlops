"""Validate the raw Telco Customer Churn Dataset"""

from pathlib import Path

import pandas as pd

from customer_churn.data.schema import (
    EXPECTED_COLUMNS,
    EXPECTED_ROW_COUNT,
    VALID_CHURN_VALUES,
)

RAW_DATA_FILE = Path("data/raw/telco_customer_churn.csv")

class DataValidationError(ValueError):
    """Raised when the dataset fails a validation check."""


def load_raw_data(path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Load the raw customer churn dataset"""

    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset was not found at: {path}"
        )
    return pd.read_csv(path)


def validate_schema(dataframe: pd.DataFrame) -> None:
    """Validate expected dataset columns"""

    actual_columns = set(dataframe.columns)
    expected_columns = set(EXPECTED_COLUMNS)

    missing_columns = sorted(expected_columns - actual_columns)
    unexpected_columns = sorted(actual_columns - expected_columns)

    if missing_columns:
        raise DataValidationError(
            f"Missing expected columns: {missing_columns}"
        )

    if unexpected_columns:
        raise DataValidationError(
            f"Unexpected columns found: {unexpected_columns}"
        )


def validate_row_count(dataframe: pd.DataFrame) -> None:
    """Validate the expected number of source records"""

    if len(dataframe) != EXPECTED_ROW_COUNT:
        raise DataValidationError(
            f"Expected {EXPECTED_ROW_COUNT} rows,"
            f"but found {len(dataframe)}."
        )


def validate_customer_ids(dataframe: pd.DataFrame) -> None:
    """Validate customer identifiers"""

    customer_ids = dataframe["customerID"]

    if customer_ids.isna().any():
        raise DataValidationError(
            "customerID contains missing values."
        )

    if customer_ids.astype(str).str.strip().eq("").any():
        raise DataValidationError(
            "customerID contains blank values."
        )

    duplicate_count = customer_ids.duplicated().sum()

    if duplicate_count > 0:
        raise DataValidationError(
            f"Found {duplicate_count} duplicate customerID values."
        )


def validate_target(dataframe: pd.DataFrame) -> None:
    """Validate the churn target variable"""

    if dataframe["Churn"].isna().any():
        raise DataValidationError(
            "Churn contains missing values."
        )

    actual_values = set(dataframe["Churn"].unique())
    invalid_values = actual_values - VALID_CHURN_VALUES

    if invalid_values:
        raise DataValidationError(
            f"Invalid Churn values found: {sorted(invalid_values)}"
        )


def validate_numeric_fields(dataframe: pd.DataFrame) -> None:
    """Validate important numeric fields"""

    if (dataframe["tenure"] < 0).any():
        raise DataValidationError(
            "Tenure contains negative values."
        )

    if (dataframe["MonthlyCharges"] < 0).any():
        raise DataValidationError(
            "MonthlyCharges contains negative values."
        )

    valid_senior_values = {0, 1}
    actual_senior_values = set(
        dataframe["SeniorCitizen"].dropna().unique()
    )

    if not actual_senior_values.issubset(valid_senior_values):
        raise DataValidationError(
            "SeniorCitizen contains values other than 0 or 1."
        )


def validate_dataframe(dataframe: pd.DataFrame) -> None:
    """Run all raw-data validation checks"""

    validate_schema(dataframe)
    validate_row_count(dataframe)
    validate_customer_ids(dataframe)
    validate_target(dataframe)
    validate_numeric_fields(dataframe)


def main() -> None:
    """Load and validate the raw dataset"""

    dataframe = load_raw_data()

    print("Validating Telco Customer Churn dataset...")

    validate_dataframe(dataframe)

    print("Data validation passed.")
    print(f"Rows: {dataframe.shape[0]}")
    print(f"Columns: {dataframe.shape[1]}")
    print(
        f"Duplicate customer IDs: "
        f"{dataframe['customerID'].duplicated().sum()}"
    )

    missing_values = dataframe.isna().sum()
    missing_values = missing_values[missing_values > 0]

    if missing_values.empty:
        print("Missing values: None")
    else:
        print("Missing values detected:")
        print(missing_values)


if __name__ == "__main__":
    main()