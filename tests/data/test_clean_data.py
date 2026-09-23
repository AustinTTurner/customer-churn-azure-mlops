"""Tests for customer churn data cleaning"""

import pandas as pd
import pytest


from customer_churn.data.clean_data import(
    clean_dataframe,
    validate_cleaned_data,
)


def test_total_charges_is_converted_to_numeric():
    dataframe = pd.DataFrame(
        {
            "TotalCharges": ["100.50", "200.25"],
            "tenure": [1, 2]
        }
    )

    cleaned = clean_dataframe(dataframe)

    assert pd.api.types.is_numeric_dtype(
        cleaned["TotalCharges"]
    )


def test_blank_total_charge_with_zero_tenure_becomes_zero():
    dataframe = pd.DataFrame(
        {
            "TotalCharges": [" "],
            "tenure": [0],
        }
    )

    cleaned = clean_dataframe(dataframe)

    assert cleaned.loc[0, "TotalCharges"] == 0.0


def test_missing_total_charges_after_cleaning_fails():
    dataframe = pd.DataFrame(
        {
            "TotalCharges": [None],
            "tenure": [12],
        }
    )

    cleaned = clean_dataframe(dataframe)

    with pytest.raises(
        ValueError,
        match = "missing values",
    ):
        validate_cleaned_data(cleaned)