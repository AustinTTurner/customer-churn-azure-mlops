"""Create a versioned customer churn model package"""

import hashlib
import json
import platform
import shutil
from pathlib import Path

import sklearn

from customer_churn.features.preprocessing import MODEL_FEATURES


MODEL_NAME = "Balanced Logistic Regression"
MODEL_VERSION = "1.0.0"

SOURCE_MODEL = Path (
    "artifacts/models/tuned_balanced_logistic_regression.joblib"
)

THRESHOLD_SUMMARY = Path (
    "reports/threshold_summary.json"
)

PACKAGE_DIR = Path (
    "artifacts/packages/customer_churn_v1"
)

REPORT_MANIFEST = Path (
    "reports/model_package_manifest.json"
)


def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 digest of a file"""

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter (
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    """Package the selected model and its metadata"""

    if not SOURCE_MODEL.is_file():
        raise FileNotFoundError (
            f"Trained model not found: {SOURCE_MODEL}"
        )

    if not THRESHOLD_SUMMARY.is_file():
        raise FileNotFoundError (
            f"Threshold summary not found: {THRESHOLD_SUMMARY}"
        )

    with THRESHOLD_SUMMARY.open (
        "r",
        encoding = "utf-8,"
    ) as file:
        threshold_summary = json.load(file)

    selected = threshold_summary[MODEL_NAME]

    threshold = float (
        selected["selected_threshold"]
    )

    recall_target = float (
        selected["recall_target"]
    )

    recall = float (
        selected["selected_metrics"]["recall"]
    )

    if not 0 <= threshold <= 1:
        raise ValueError (
            "Decision threshold must be between 0 and 1."
        )

    if (
        not selected["recall_target_met"]
        or recall < recall_target
    ):
        raise ValueError (
            "Selected model does not meet the recall target."
        )

    PACKAGE_DIR.mkdir (
        parents = True,
        exist_ok = True,
    )

    packaged_model = PACKAGE_DIR / "model.joblib"

    shutil.copy2 (
        SOURCE_MODEL,
        packaged_model,
    )

    manifest = {
        "schema_version": 1,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_filename": "model.joblib",
        "model_sha256": calculate_sha256 (
            packaged_model
        ),
        "decision_threshold": threshold,
        "positive_class": 1,
        "positive_class_label": "Yes",
        "score_calibrated": False,
        "feature_columns": MODEL_FEATURES,
        "training_data": {
            "dataset": "IBM Telco Customer Churn",
            "openml_dataset_id": 42178,
            "split": "stratified 80/20",
            "random_state": 42,
        },
        "development_evaluation": {
            "method": "training-only out-of-fold predictions",
            "recall_target": recall_target,
            "recall_target_met": True,
            "selected_metrics": selected [
                "selected_metrics"
            ],
            "limitations": [
                "Hyperparameters and the threshold were selected using the same training dataset.",
                "The original test set was examined during earlier model comparison.",
                "An independent dataset is needed for an unbiased final performance estimate."
            ],
        },
        "environment": {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
        },
    }

    manifest_text = (
        json.dumps(manifest, indent = 4) + "\n"
    )

    package_manifest = (
        PACKAGE_DIR / "manifest.json"
    )

    package_manifest.write_text (
        manifest_text,
        encoding = "utf-8",
    )

    REPORT_MANIFEST.parent.mkdir (
        parents = True,
        exist_ok = True,
    )

    REPORT_MANIFEST.write_text (
        manifest_text,
        encoding = "utf-8",
    )

    print("Model package created successfully.")
    print(f"Model: {MODEL_NAME}")
    print(f"Version: {MODEL_VERSION}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Package: {PACKAGE_DIR}")
    print(f"Manifest: {REPORT_MANIFEST}")


if __name__ == "__main__":
    main()