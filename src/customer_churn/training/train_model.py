"""Train the baseline customer churn classification model"""

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from customer_churn.features.prepare_features import (
    load_processed_data,
    prepare_features,
    split_data,
)

from customer_churn.features.preprocessing import (
    build_preprocessor,
)


MODEL_DIR = Path("artifacts/models")
MODEL_FILE = MODEL_DIR / "baseline_logistic_regression.joblib"

RANDOM_STATE = 42


def build_model_pipeline() -> Pipeline:
    """Create the full preprocessing and classification pipeline"""

    preprocessor = build_preprocessor()

    classifier = LogisticRegression(
        max_iter = 1000,
        random_state = RANDOM_STATE,
    )

    model_pipeline = Pipeline(
        steps = [
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

    return model_pipeline


def train_model():
    """Train the baseline churn model"""

    dataframe = load_processed_data()

    X, y = prepare_features(dataframe)

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
    )

    model = build_model_pipeline()

    print("Training baseline Logistic Regression model...")

    model.fit(
        X_train,
        y_train,
    )

    print("Training completed.")

    return (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
    )


def save_model (
        model: Pipeline,
        path: Path = MODEL_FILE,
) -> Path:
    """Save the trained model pipeline."""

    MODEL_DIR.mkdir(
        parents = True,
        exist_ok = True,
    )

    joblib.dump(
        model,
        path,
    )

    return path


def main() -> None:
    """Train and save the baseline model"""

    (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_model()

    model_path = save_model(model)

    print()
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()