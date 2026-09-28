"""Compare the churn model with a naive baseline"""

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from customer_churn.features.prepare_features import (
    load_processed_data,
    prepare_features,
    split_data,
)

from customer_churn.training.train_model import (
    build_model_pipeline,
)


def main() -> None:
    """Compare Logistic Regression with a majority baseline"""

    dataframe = load_processed_data()

    X, y = prepare_features(dataframe)

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
    )

    dummy_model = DummyClassifier(
        strategy = "most_frequent",
    )

    dummy_model.fit(
        X_train,
        y_train,
    )

    logistic_model = build_model_pipeline()

    logistic_model.fit(
        X_train,
        y_train,
    )

    models = {
        "Majority Baseline": dummy_model,
        "Logistic Regression": logistic_model,
    }

    for name, model in models.items():

        predictions = model.predict(
            X_test
        )

        print()
        print(name)
        print("-" * len(name))

        print(
            f"Accuracy: "
            f"{accuracy_score(y_test, predictions):.4f}"
        )

        print(
            f"Precision: "
            f"{precision_score(y_test, predictions, zero_division = 0):.4f}"
        )

        print(
            f"Recall: "
            f"{recall_score(y_test, predictions, zero_division = 0):.4f}"
        )

        print(
            f"F1: "
            f"{f1_score(y_test, predictions, zero_division = 0)}"
        )


if __name__ == "__main__":
    main()
