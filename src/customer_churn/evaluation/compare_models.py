"""Compare candidate customer churn classification models"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.pipeline import Pipeline

from customer_churn.features.prepare_features import (
    load_processed_data,
    prepare_features,
    split_data,
)

from customer_churn.features.preprocessing import (
    build_preprocessor,
)


RANDOM_STATE = 42

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

COMPARISON_FILE = REPORTS_DIR / "model_comparison.csv"
COMPARISON_FIGURE = FIGURES_DIR / "model_comparison.png"


def build_candidate_models() -> dict[str, Pipeline]:
    """Create candidate model pipelines"""

    models = {
        "Logistic Regression": Pipeline (
            steps = [
                (
                    "preprocessor",
                    build_preprocessor(),
                ),
                (
                    "classifier",
                    LogisticRegression (
                        max_iter = 1000,
                        random_state = RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "Balanced Logistic Regression": Pipeline (
            steps = [
                (
                    "preprocessor",
                    build_preprocessor(),
                ),
                (
                    "classifier",
                    LogisticRegression (
                        max_iter = 1000,
                        class_weight = "balanced",
                        random_state = RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "Random Forest": Pipeline (
            steps = [
                (
                    "preprocessor",
                    build_preprocessor(),
                ),
                (
                    "classifier",
                    RandomForestClassifier (
                        n_estimators = 300,
                        class_weight = "balanced",
                        random_state = RANDOM_STATE,
                        n_jobs = -1,
                    ),
                ),
            ]
        ),

        "Gradient Boosting": Pipeline (
            steps = [
                (
                    "preprocessor",
                    build_preprocessor(),
                ),
                (
                    "classifier",
                    GradientBoostingClassifier (
                        random_state = RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    return models


def evaluate_model (
        model,
        X_test,
        y_test,
) -> dict[str, float]:
    """Calculate evaluation metrics for one candidate model"""

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    return {
        "accuracy": accuracy_score (
            y_test,
            predictions,
        ),
        "precision": precision_score (
            y_test,
            predictions,
            zero_division = 0,
        ),
        "recall": recall_score (
            y_test,
            predictions,
            zero_division = 0,
        ),
        "f1": f1_score (
            y_test,
            predictions,
            zero_division = 0,
        ),
        "roc_auc": roc_auc_score (
            y_test,
            probabilities,
        ),
    }


def compare_models() -> pd.DataFrame:
    """Train and compare all condidate models"""

    dataframe = load_processed_data()

    X, y = prepare_features(dataframe)

    X_train, X_test, y_train, y_test = split_data (
        X,
        y,
    )

    candidate_models = build_candidate_models()

    results = []

    for name, model in candidate_models.items():

        print()
        print(f"Training: {name}")

        model.fit (
            X_train,
            y_train,
        )

        metrics = evaluate_model (
            model,
            X_test,
            y_test,
        )

        results.append (
            {
                "model": name,
                **metrics,
            }
        )

        print(
            f"Accuracy : {metrics['accuracy']:.4f}"
        )
        print(
            f"Precision: {metrics['precision']:.4f}"
        )
        print(
            f"Recall   : {metrics['recall']:.4f}"
        )
        print(
            f"F1       : {metrics['f1']:.4f}"
        )
        print(
            f"ROC-AUC  : {metrics['roc_auc']:.4f}"
        )

    results_dataframe = pd.DataFrame (
        results
    )

    return results_dataframe


def save_results (
        results: pd.DataFrame,
) -> Path:
    """Save model comparison results"""

    REPORTS_DIR.mkdir (
        parents = True,
        exist_ok = True,
    )

    results.to_csv (
        COMPARISON_FILE,
        index = False,
    )

    return COMPARISON_FILE


def save_comparison_chart (
        results: pd.DataFrame,
) -> Path:
    """Save model metric comparison chart"""

    FIGURES_DIR.mkdir (
        parents = True,
        exist_ok = True,
    )

    plot_data = results.set_index (
        "model"
    )[
        [
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ]
    ]

    plot_data.plot (
        kind = "bar",
        figsize = (11, 6),
    )

    plt.title (
        "Customer Churn Model Comparison"
    )
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.ylim(0, 1)
    plt.xticks (
        rotation = 25,
        ha = "right",
    )
    plt.legend (
        title = "Metric"
    )
    plt.tight_layout()

    plt.savefig (
        COMPARISON_FIGURE,
        dpi = 150,
    )

    plt.close()

    return COMPARISON_FIGURE


def main() -> None:
    """Run candidate model comparison"""

    results = compare_models()

    print()
    print("Model Comparison")
    print("=" * 80)

    print(
        results.to_string (
            index = False,
            float_format = lambda value: f"{value:.4f}",
        )
    )

    results_path = save_results (
        results
    )

    figure_path = save_comparison_chart (
        results
    )

    print()
    print(
        f"Comparison results saved to: "
        f"{results_path}"
    )
    print(
        f"Comparison chart saved to: "
        f"{figure_path}"
    )


if __name__ == "__main__":
    main()
