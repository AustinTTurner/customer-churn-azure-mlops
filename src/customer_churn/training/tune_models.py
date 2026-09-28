"""Tune churn classification models using cross-validation"""

import json
from pathlib import Path

import joblib 
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from customer_churn.evaluation.compare_models import (
    build_candidate_models,
)

from customer_churn.features.prepare_features import (
    load_processed_data,
    prepare_features,
    split_data,
)

RANDOM_STATE = 42

REPORTS_DIR = Path("reports")
MODEL_DIR = Path("artifacts/models")

SUMMARY_FILE = REPORTS_DIR / "tuning_summary.json"

SCORING = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def build_cross_validator() -> StratifiedKFold:
    """Create reproducible, stratified CV folds"""

    return StratifiedKFold (
        n_splits = 5,
        shuffle = True,
        random_state = RANDOM_STATE,
    )


def get_tuning_configs() -> dict:
    """Define the condidate models and search spaces"""

    return {
        "Balanced Logistic Regression": {
            "filename": "tuned_balanced_logistic_regression",
            "param_grid": {
                "classifier__C": [0.1, 1.0, 10.0],
            },
        },
        "Gradient Boosting": {
            "filename": "tuned_gradient_boosting",
            "param_grid": {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [2, 3],
            },
        },
    }


def build_search (
        model: Pipeline,
        param_grid: dict,
) -> GridSearchCV:
    """Create a multi-metric hyperparameter search"""

    return GridSearchCV (
        estimator = model,
        param_grid = param_grid,
        scoring = SCORING,
        refit = "f1",
        cv = build_cross_validator(),
        n_jobs = 1,
        verbose = 1,
        error_score = "raise",
    )


def main() -> None:
    """Tune candidates using training data only"""

    dataframe = load_processed_data()

    X, y = prepare_features(dataframe)

    X_train, _, y_train, _ = split_data(X, y)

    models = build_candidate_models()
    configs = get_tuning_configs()

    REPORTS_DIR.mkdir (
        parents = True,
        exist_ok = True,
    )

    MODEL_DIR.mkdir (
        parents = True,
        exist_ok = True,
    )

    summary = {}

    print(f"Training records: {len(X_train)}")
    print(f"Test set excluded from tuning.")

    for name, config in configs.items():
        print()
        print("=" * 50)
        print(f"Tuning: {name}")
        print("=" * 50)

        search = build_search (
            model = models[name],
            param_grid = config["param_grid"],
        )

        search.fit(X_train, y_train)

        best_index = search.best_index_
        results = search.cv_results_

        cv_metrics = {
            metric: float (
                results[f"mean_test_{metric}"][best_index]
            )
            for metric in SCORING
        }

        cv_f1_std = float (
            results["std_test_f1"][best_index]
        )

        summary[name] = {
            "best_params": search.best_params_,
            "cv_metrics": cv_metrics,
            "cv_f1_std": cv_f1_std,
        }

        print()
        print("Best parameters:")
        print(search.best_params_)

        print()
        print("Cross-validation results:")

        for metric, value in cv_metrics.items():
            print(f"{metric}: {value:.4f}")

        print(f"F1 standard deviation: {cv_f1_std:.4f}")

        filename = config["filename"]

        # Save every configuration's CV results
        results_dataframe = pd.DataFrame (
            search.cv_results_
        )

        results_file = (
            REPORTS_DIR / f"{filename}_cv_results.csv"
        )

        results_dataframe.to_csv (
            results_file,
            index = False,
        )

        # Save the best fitted pipeline for this model
        model_file = (
            MODEL_DIR / f"{filename}.joblib"
        )

        joblib.dump (
            search.best_estimator_,
            model_file,
        )

        print(f"CV results saved to: {results_file}")
        print(f"Model saved to: {model_file}")

    with SUMMARY_FILE.open (
        "w",
        encoding = "utf-8",
    ) as file:
        json.dump (
            summary,
            file,
            indent = 4,
        )

    print()
    print("Hyperparameter tuning completed.")
    print(f"Summary saved to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()