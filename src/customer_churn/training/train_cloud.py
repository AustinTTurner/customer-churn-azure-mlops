"""Train the selected churn model in Azure Machine Learning"""

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)

from sklearn.pipeline import Pipeline

from customer_churn.features.prepare_features import (
    prepare_features,
    split_data,
)

from customer_churn.features.preprocessing import (
    MODEL_FEATURES,
    build_preprocessor,
)

RANDOM_STATE = 42
DECISION_THRESHOLD = 0.55
LOGISTIC_C = 0.1


def build_cloud_model() -> Pipeline:
    """Build the selected deployment candidate"""

    return Pipeline (
        steps = [
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                LogisticRegression (
                    C = LOGISTIC_C,
                    class_weight = "balanced",
                    max_iter = 1000,
                    random_state = RANDOM_STATE,
                ),
            ),
        ]
    )


def calculate_oof_metrics (
        model: Pipeline,
        X_train,
        y_train,
) -> dict:
    """Calculate development metrics from OOF predictions"""

    cv = StratifiedKFold (
        n_splits = 5,
        shuffle = True,
        random_state = RANDOM_STATE,
    )

    probabilities = cross_val_predict (
        model,
        X_train,
        y_train,
        cv = cv,
        method = "predict_proba",
        n_jobs = 1,
    )[:, 1]

    predictions = (
        probabilities >= DECISION_THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix (
        y_train,
        predictions,
        labels = [0, 1],
    ).ravel()

    return {
        "accuracy_oof": accuracy_score (
            y_train,
            predictions,
        ),
        "precision_oof": precision_score (
            y_train,
            predictions,
            zero_division = 0,
        ),
        "recall_oof": recall_score (
            y_train,
            predictions,
            zero_division = 0,
        ),
        "f1_oof": f1_score (
            y_train,
            predictions,
            zero_division = 0,
        ),
        "f2_oof": fbeta_score (
            y_train,
            predictions,
            beta = 2,
            zero_division = 0,
        ),
        "roc_auc_oof": roc_auc_score (
            y_train,
            probabilities,
        ),
        "tp_oof": int(tp),
        "fp_oof": int(fp),
        "tn_oof": int(tn),
        "fn_oof": int(fn),
    }


def main() -> None:
    """Train, log, and save the final cloud model"""

    parser = argparse.ArgumentParser()

    parser.add_argument (
        "--data",
        required = True,
        help = "Path to cleaned churn CSV."
    )

    parser.add_argument (
        "--model-output",
        required = True,
        help = "Azure ML output directory.",
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.model_output)

    dataframe = pd.read_csv(data_path)

    X, y = prepare_features(dataframe)

    X_train, _, y_train, _ = split_data(X, y)

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
    mlflow.log_param ("C", LOGISTIC_C)
    mlflow.log_param (
        "class_weight",
        "balanced",
    )
    mlflow.log_param (
        "decision_threshold",
        DECISION_THRESHOLD,
    )
    mlflow.log_param (
        "random_state",
        RANDOM_STATE,
    )

    for name, value in metrics.items():
        mlflow.log_metric (
            name,
            float(value),
        )

    print()
    print("Development OOF metrics")

    for name, value in metrics.items():
        print(f"{name}: {value}")

    # Train the deployment model on all available cleaned data
    model.fit(X, y)

    output_dir.mkdir (
        parents = True,
        exist_ok = True,
    )

    model_path = output_dir / "model.joblib"

    joblib.dump (
        model,
        model_path,
    )

    manifest = {
        "model_name": "Balanced Logistic Regression",
        "model_version": "cloud-v1",
        "decision_threshold": DECISION_THRESHOLD,
        "classifier_C": LOGISTIC_C,
        "class_weight": "balanced",
        "feature_columns": MODEL_FEATURES,
        "training_records": int(len(X)),
        "development_cv_records": int(len(X_train)),
        "scikit_learn_version": sklearn.__version__,
        "metrics": metrics,
        "evaluation_note": (
            "Metrics are development-stage five-fold out-of-fold estimates using the original training partition."
            "They are not an independent final evaluation."
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

    mlflow.log_artifact (
        str(manifest_path),
        artifact_path = "metadata",
    )

    print()
    print("Cloud model training completed.")
    print(f"Model saved to: {model_path}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()