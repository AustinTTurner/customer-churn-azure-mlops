"""Train the selected churn model inside an Azure ML pipeline"""

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import sklearn

from customer_churn.features.prepare_features import (
    prepare_features,
    split_data,
)
from customer_churn.features.preprocessing import (
    MODEL_FEATURES,
)
from customer_churn.training.train_cloud import (
    DECISION_THRESHOLD,
    LOGISTIC_C,
    build_cloud_model,
    calculate_oof_metrics,
)


def main() -> None:
    """Train and persist the pipeline model artifact"""

    parser = argparse.ArgumentParser()

    parser.add_argument (
        "--validated-data",
        required = True,
        help = "Directory containing validated data.",
    )

    parser.add_argument (
        "--model-output",
        required = True,
        help = "Directory for model artifacts.",
    )

    args = parser.parse_args()

    validated_dir = Path (
        args.validated_data
    )

    data_path = (
        validated_dir / "validated_data.csv"
    )

    output_dir = Path (
        args.model_output
    )

    if not data_path.is_file():
        raise FileNotFoundError (
            f"Validated dataset not found: "
            f"{data_path}"
        )

    dataframe = pd.read_csv (
        data_path
    )

    X, y = prepare_features (
        dataframe
    )

    X_train, _, y_train, _ = split_data (
        X,
        y,
    )

    model = build_cloud_model()

    metrics = calculate_oof_metrics (
        model,
        X_train,
        y_train,
    )

    mlflow.log_param (
        "model_type",
        "Balanced Logistic Regression",
    )

    mlflow.log_param (
        "C",
        LOGISTIC_C,
    )

    mlflow.log_param (
        "class_weight",
        "balanced",
    )

    mlflow.log_param (
        "decision_threshold",
        DECISION_THRESHOLD,
    )

    for name, value in metrics.items():
        mlflow.log_metric (
            name,
            float(value),
        )

    print()
    print("Pipeline development OOF metrics")

    for name, value in metrics.items():
        print(f"{name}: {value}")

    model.fit (
        X,
        y,
    )

    output_dir.mkdir (
        parents = True,
        exist_ok = True,
    )

    model_path = (
        output_dir / "model.joblib"
    )

    joblib.dump (
        model,
        model_path,
    )

    manifest = {
        "model_name": (
            "Balanced Logistic Regression"
        ),
        "model_version": "pipeline-v1",
        "decision_threshold": (
            DECISION_THRESHOLD
        ),
        "classifier_C": LOGISTIC_C,
        "class_weight": "balanced",
        "feature_columns": MODEL_FEATURES,
        "training_records": int (
            len(X)
        ),
        "development_cv_records": int (
            len(X_train)
        ),
        "scikit_learn_version": (
            sklearn.__version__
        ),
        "metrics": metrics,
        "evaluation_note": (
            "Metrics are development-stage five-fold out-of-fold estimates using the original training "
            "partition. They are not an independent final evaluation."
        ),
    }

    manifest_path = (
        output_dir / "manifest.json"
    )

    manifest_path.write_text (
        json.dumps (
            manifest,
            indent = 4,
        ),
        encoding = "utf-8",
    )

    print()
    print("Pipeline model training completed.")
    print(f"Model: {model_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()