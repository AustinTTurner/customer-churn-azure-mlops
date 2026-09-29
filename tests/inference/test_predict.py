"""Tests for packaged model inference"""

import json

import pandas as pd
import pytest

from sklearn.dummy import DummyClassifier

from customer_churn.inference.predict import (
    load_bundle,
    predict_customers,
)


def test_missing_required_features():
    """Reject input that lacks a required feature"""

    customers = pd.DataFrame (
        {
            "tenure": [12],
        }
    )

    manifest = {
        "feature_columns": [
            "tenure",
            "MonthlyCharges",
        ],
        "decision_threshold": 0.55,
        "positive_class": 1,
    }

    with pytest.raises (
        ValueError,
        match = "Missing required features",
    ):
        predict_customers (
            None,
            manifest,
            customers,
        )


def test_prediction_uses_manifest_threshold():
    """Apply the packaged threshold, not 0,50"""

    X_train = pd.DataFrame (
        {
            "tenure": [1, 2, 3, 4],
        }
    )

    y_train = [0, 1, 1, 1]

    model = DummyClassifier (
        strategy = "prior"
    )

    model.fit(X_train, y_train)

    customers = pd.DataFrame (
        {
            "customerID": ["example-001"],
            "tenure": [5],
        }
    )

    manifest = {
        "feature_columns": ["tenure"],
        "decision_threshold": 0.88,
        "positive_class": 1,
    }

    result = predict_customers (
        model,
        manifest,
        customers,
    )

    assert result.loc[0, "predicted_churn"] == 0

    assert result.loc[0, "customerID"] == "example-001"

    manifest["decision_threshold"] = 0.70

    result = predict_customers (
        model,
        manifest,
        customers,
    )

    assert result.loc[0, "predicted_churn"] == 1


def test_rejects_modified_model_file(tmp_path):
    """Reject a model whose SHA-256 digest is incorrect"""

    model_path = tmp_path / "model.joblib"

    model_path.write_bytes (
        b"modified-model-file"
    )

    manifest = {
        "model_filename": "model.joblib",
        "model_sha256": "0" * 64,
    }

    manifest_path = tmp_path /"manifest.json"

    manifest_path.write_text (
        json.dumps(manifest),
        encoding = "utf-8",
    )

    with pytest.raises (
        ValueError,
        match = "does not match",
    ):
        load_bundle(tmp_path)