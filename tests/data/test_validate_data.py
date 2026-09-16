"""Tests for customer churn data validation"""

import pandas as pd
import pytest

from customer_churn.data.schema import EXPECTED_COLUMNS
from customer_churn.data.validate_data import (
    DataValidationError,
    validate_dataframe,
    validate_schema,
    validate_customer_ids,
    validate_target,
)


def create_valid_dataframe() -> pd.DataFrame:
    """Create a minimal valid dataframe for unit testing"""

    row = {column: "example" for column in EXPECTED_COLUMNS}

    row.update(
        {
            "customerID": "0001-TEST",
            "SeniorCitizen": 0,
            "tenure": 12,
            "MonthlyCharges": 50.0,
            "TotalCharges": 600.0,
            "Churn": "No",
        }
    )

    dataframe = pd.DataFrame([row] * 7043)

    dataframe["customerID"] = [
        f"CUSTOMER-{index:04d}"
        for index in range(7043)
    ]

    return dataframe


def test_valid_dataframe_passes():
    dataframe = create_valid_dataframe()

    validate_dataframe(dataframe)


def test_missing_column_fails():
    dataframe = create_valid_dataframe()
    dataframe = dataframe.drop(columns = ["tenure"])

    with pytest.raises(
        DataValidationError,
        match = "Missing expected columns",
    ):
        validate_schema(dataframe)


def test_duplicate_customer_id_fails():
    dataframe = create_valid_dataframe()

    dataframe.loc[1, "customerID"] = dataframe.loc[0, "customerID"]

    with pytest.raises(
        DataValidationError,
        match = "duplicate customerID",
    ):
        validate_customer_ids(dataframe)


def test_invalid_churn_value_fails():
    dataframe = create_valid_dataframe()

    dataframe.loc[0, "Churn"] = "Maybe"

    with pytest.raises(
        DataValidationError,
        match = "Invalid Churn values",
    ):
        validate_target(dataframe)