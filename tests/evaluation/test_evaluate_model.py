"""Tests for model evaluation"""

import pandas as pd

from customer_churn.evaluation.evaluate_model import (
    calculate_metrics,
)

from customer_churn.training.train_model import (
    build_model_pipeline,
)


def create_training_data():
    """Create a small valid training dataset"""

    dataframe = pd.DataFrame (
        {
            "SeniorCitizen": [0, 1, 0, 1],
            "tenure": [60, 2, 48, 4],
            "MonthlyCharges": [
                40.0,
                95.0,
                50.0,
                90.0,
            ],
            "TotalCharges": [
                2400.0,
                190.0,
                2400.0,
                360.0,
            ],
            "gender": [
                "Female",
                "Male",
                "Male",
                "Female",
            ],
            "Partner": [
                "Yes",
                "No",
                "Yes",
                "No",
            ],
            "Dependents": [
                "Yes",
                "No",
                "Yes",
                "No",
            ],
            "PhoneService": ["Yes"] * 4,
            "MultipleLines": [
                "No",
                "Yes",
                "No",
                "Yes",
            ],
            "InternetService": [
                "DSL",
                "Fiber optic",
                "DSL",
                "Fiber optic",
            ],
            "OnlineSecurity": [
                "Yes",
                "No",
                "Yes",
                "No",
            ],
            "OnlineBackup": [
                "Yes",
                "No",
                "Yes",
                "No",
            ],
            "DeviceProtection": [
                "Yes",
                "No",
                "Yes",
                "No",
            ],
            "TechSupport": [
                "Yes",
                "No",
                "Yes",
                "No",
            ],
            "StreamingTV": [
                "No",
                "Yes",
                "No",
                "Yes",
            ],
            "StreamingMovies": [
                "No",
                "Yes",
                "No",
                "Yes",
            ],
            "Contract": [
                "Two year",
                "Month-to-month",
                "One-year",
                "Month-to-month",
            ],
            "PaperlessBilling": [
                "No",
                "Yes",
                "No",
                "Yes",
            ],
            "PaymentMethod": [
                "Credit card (automatic)",
                "Electronic check",
                "Bank transfer (automatic)",
                "Electronic check",
            ],
        }
    )

    target = pd.Series(
        [0, 1, 0, 1]
    )

    return dataframe, target


def test_calculate_metrics_returns_expected_keys():
    X, y = create_training_data()

    model = build_model_pipeline()

    model.fit(
        X,
        y,
    )

    metrics = calculate_metrics (
        model,
        X,
        y,
    )

    expected_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }

    assert set(metrics.keys()) == expected_metrics