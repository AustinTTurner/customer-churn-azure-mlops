"""Load the packaged model and predict customer churn"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from customer_churn.features.preprocessing import ID_COLUMN
from customer_churn.packaging.package_model import (
    calculate_sha256,
)

DEFAULT_PACKAGE = Path (
    "artifacts/packages/customer_churn_v1"
)


def load_bundle (
        package_dir: Path = DEFAULT_PACKAGE,
):
    """Load and verify a trusted local model package"""

    package_dir = Path(package_dir)

    manifest_path = package_dir / "manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError (
            f"Package manifest not found: {manifest_path}"
        )

    with manifest_path.open (
        "r",
        encoding = "utf-8",
    ) as file:
        manifest = json.load(file)

    if manifest["model_filename"] != "model.joblib":
        raise ValueError (
            "Unexpected model filename in manifest."
        )

    model_path = package_dir / "model.joblib"

    if not model_path.is_file():
        raise FileNotFoundError (
            f"Package model not found: {model_path}"
        )

    actual_hash = calculate_sha256(model_path)

    if actual_hash != manifest["model_sha256"]:
        raise ValueError (
            "Model file does not match the package manifest."
        )

    trained_version = manifest [
        "environment"
    ]["scikit_learn_version"]

    if trained_version != sklearn.__version__:
        raise RuntimeError (
            "skikit-learn version differs from the model's recorded training environmnet."
        )


    # Only load packages created by our trusted project
    model = joblib.load(model_path)

    return model, manifest


def predict_customers (
        model,
        manifest: dict,
        customers: pd.DataFrame,
) -> pd.DataFrame:
    """Score customers using the packaged model."""

    if customers.empty:
        raise ValueError (
            "At least one customer record is required."
        )

    required_features = manifest["feature_columns"]

    missing_features = [
        column
        for column in required_features
        if column not in customers.columns
    ]

    if missing_features:
        raise ValueError (
            "Missing required features: "
            + ", ".join(missing_features)
        )

    threshold = float (
        manifest["decision_threshold"]
    )

    if not 0 <= threshold <= 1:
        raise ValueError (
            "Decision threshold must be between 0 an 1."
        )

    # Exclude customerIDm churn and any extra columns from the model input.
    X = customers.loc [
        :,
        required_features,
    ]

    positive_class = manifest["positive_class"]

    classes = list(model.classes_)

    if positive_class not in classes:
        raise ValueError (
            "Positive class not found in the model."
        )

    positive_index = classes.index (
        positive_class
    )

    scores = model.predict_proba(X) [
        :,
        positive_index,
    ]

    predictions = (
        scores >= threshold
    ).astype(int)

    output = pd.DataFrame (
        {
            "churn_score": scores,
            "predicted_churn": predictions,
            "decision_threshold": threshold,
        }
    )

    if ID_COLUMN in customers.columns:
        output.insert (
            0,
            ID_COLUMN,
            customers[ID_COLUMN].to_numpy(),
        )

    return output



def main() -> None:
    """Run a local inference demonstration."""

    parser = argparse.ArgumentParser (
        description="Predict customer churn."
    )

    parser.add_argument (
        "--input",
        type=Path,
        required=True,
        help="CSV containing customer features.",
    )

    parser.add_argument (
        "--limit",
        type=int,
        default=5,
        help="Number of records to score.",
    )

    args = parser.parse_args()

    if args.limit < 1:
        parser.error (
            "--limit must be at least 1."
        )

    customers = pd.read_csv (
        args.input,
        nrows=args.limit,
    )

    model, manifest = load_bundle()

    predictions = predict_customers (
        model,
        manifest,
        customers,
    )

    print()
    print(f"Model: {manifest['model_name']}")
    print(f"Version: {manifest['model_version']}")
    print(
        "Decision threshold: "
        f"{manifest['decision_threshold']:.2f}"
    )

    print()
    print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()
