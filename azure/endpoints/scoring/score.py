"""Azure ML real-time scoring entry point for customer churn"""

import json
import os
from pathlib import Path

import joblib
import pandas as pd


MODEL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

model = None
decision_threshold = None


def _find_artifact (
        model_directory: Path,
        filename: str,
) -> Path:
    """Locate exactly one requested artifact"""

    matches = list (
        model_directory.rglob(filename)
    )

    if not matches:
        raise FileNotFoundError (
            f"{filename} was not found beneath "
            f"{model_directory}" 
        )

    if len(matches) > 1:
        raise RuntimeError (
            f"Multiple {filename} files were found: "
            f"{matches}"
        )

    return matches[0]


def init() -> None:
    """Load the model and deployment metadata"""

    global model
    global decision_threshold

    model_directory = Path (
        os.environ["AZUREML_MODEL_DIR"]
    )

    model_path = _find_artifact (
        model_directory,
        "model.joblib",
    )

    manifest_path = _find_artifact (
        model_directory,
        "manifest.json",
    )

    manifest = json.loads (
        manifest_path.read_text (
            encoding = "utf-8"
        )
    )

    decision_threshold = float (
        manifest["decision_threshold"]
    )

    model = joblib.load (
        model_path
    )

    print(
        "Customer churn model loaded successfully."
    )

    print(
        f"Decision threshold: "
        f"{decision_threshold}"
    )


def run(raw_data):
    """Score one or more customer records"""

    if model is None:
        raise RuntimeError (
            "Model has not been initalized."
        )

    if isinstance(raw_data, str):
        payload = json.loads(raw_data)
    else:
        payload = raw_data

    records = payload.get("records")

    if not isinstance(records, list) or not records:
        raise ValueError (
            "Request must contain a non-emtpy 'records' list."
        )

    dataframe = pd.DataFrame (
        records
    )

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in dataframe.columns
    ]

    if missing_features:
        raise ValueError (
            "Missing required features: "
            f"{missing_features}"
        )

    dataframe = dataframe [
        MODEL_FEATURES
    ]

    scores = model.predict_proba (
        dataframe
    )[:, 1]

    predictions = (
        scores >= decision_threshold
    ).astype(int)

    results =[]

    for score, prediction in zip (
        scores,
        predictions,
    ):
        results.append (
            {
                "churn_score": float(score),
                "predicted_churn": int (
                    prediction
                ),
                "predicted_label": (
                    "Yes"
                    if prediction == 1
                    else "No"
                ),
                "decision_threshold": float (
                    decision_threshold
                ),
            }
        )

    return {
        "predictions": results
    }
