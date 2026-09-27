"""Tests for feature preprocessing"""

import pandas as pd

from customer_churn.features.preprocessing import (
    MODEL_FEATURES,
    build_preprocessor,
)

from customer_churn.features.prepare_features import (
    prepare_features,
    split_data,
)


def create_sample_dataframe() -> pd.DataFrame:
    """Create sample churn records for testing"""

    return pd.DataFrame(
        [
            {
                "customerID": "CUSTOMER-001",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 55.0,
                "TotalCharges": 660.0,
                "Churn": "No",
            },
            {
                "customerID": "CUSTOMER-002",
                "gender": "Male",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 95.0,
                "TotalCharges": 190.0,
                "Churn": "Yes",
            },
        ]
    )


def test_customer_id_is_not_model_features():
    assert "customerID" not in MODEL_FEATURES


def test_churn_is_not_model_feature():
    assert "Churn" not in MODEL_FEATURES


def test_prepare_features_encodes_target():
    dataframe = create_sample_dataframe()

    X, y = prepare_features(dataframe)

    assert y.tolist() == [0, 1]

    assert "customerID" not in X.columns
    assert "Churn" not in X.columns


def test_preprocessor_transforms_features():
    dataframe = create_sample_dataframe()

    X, _ = prepare_features(dataframe)

    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == 2
    assert transformed.shape[1] > len(MODEL_FEATURES)


def test_preprocessor_handles_unknown_category():
    dataframe = create_sample_dataframe()

    X, _ = prepare_features(dataframe)

    preprocessor = build_preprocessor()
    preprocessor.fit(X)

    new_customer = X.iloc[[0]].copy()

    new_customer.loc[
        new_customer.index[0],
        "PaymentMethod",
    ] = "New payment method"

    transformed = preprocessor.transform(
        new_customer
    )

    assert transformed.shape[0] == 1