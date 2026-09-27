"""Prepare customer churn data for machine-learning training"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from customer_churn.features.preprocessing import (
    ID_COLUMN,
    MODEL_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
)

PROCESSED_DATA_FILE = Path (
    "data/processed/telco_customer_churn_clean.csv"
)

TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_processed_data (
        path: Path = PROCESSED_DATA_FILE,
) -> pd.DataFrame:
    """Load the cleaned customer churn dataset"""

    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset was not found at: {path}"
        )
    return pd.read_csv(path)


def prepare_features (
        dataframe: pd.DataFrame,
):
    """Separate model features from the target variable"""

    X = dataframe[MODEL_FEATURES].copy()

    y = (
        dataframe[TARGET_COLUMN]
        .map(
            {
                "No": 0,
                "Yes": 1,
            }
        )
    )

    if y.isna().any():
        raise ValueError(
            "Unexpected values were found in the Churn target."
        )

    return X, y.astype(int)


def split_data (
        X: pd.DataFrame,
        y: pd.Series,
):
    """Split data into stratified train and test datasets"""

    return train_test_split (
        X,
        y,
        test_size = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify = y,
    )


def main() -> None:
    """Prepare and verify model-ready features"""

    dataframe = load_processed_data()

    print("Preparing model features...")

    print(f"Original rows: {dataframe.shape[0]}")
    print(f"Original columns: {dataframe.shape[1]}")

    X, y = prepare_features(dataframe)

    X_train, X_test, y_train, y_test = split_data (
        X,
        y,
    )

    preprocessor = build_preprocessor()

    X_train_transformed = preprocessor.fit_transform(
        X_train
    )

    X_test_transformed = preprocessor.transform(
        X_test
    )

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    print()
    print("Feature preparation completed.")
    print(f"Model input features: {X.shape[1]}")
    print(f"Training rows: {X_train.shape[0]}")
    print(f"Testing rows: {X_test.shape[0]}")
    print(
        "Transformed training shape:",
        X_train_transformed.shape,
    )
    print(
        "Transformed testing shape:",
        X_test_transformed.shape,
    )

    print()
    print(
        f"Final transformed feature count: "
        f"{len(feature_names)}"
    )

    print()
    print("First 10 transformed feautres:")

    for feature in feature_names[:10]:
        print(f" - {feature}")

    print()
    print("Training churn distribution:")
    print(
        y_train.value_counts(
            normalize = True
        ).sort_index()
    )

    print()
    print("Testing churn distribution:")
    print(
        y_test.value_counts(
            normalize = True
        ).sort_index()
    )

    print()
    print(
        f"{ID_COLUMN} included as model feature:",
        ID_COLUMN in X.columns,
    )


if __name__ == "__main__":
    main()