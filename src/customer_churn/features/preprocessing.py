""" Feature preprocessing for the customer churn model"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
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
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Create the preprocessing pipeline used before model training"""

    numeric_pipeline = Pipeline(
        steps = [
            (
                "imputer",
                SimpleImputer(strategy = "median")
            ),
            (
                "scaler",
                StandardScaler(),
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps = [
            (
                "imputer",
                SimpleImputer(strategy = "most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown = "ignore",
                    sparse_output = False,
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers = [
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder = "drop",
    )

    return preprocessor