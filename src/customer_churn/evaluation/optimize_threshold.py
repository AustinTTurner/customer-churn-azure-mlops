"""Optimize churn classification decision thresholds"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)

from sklearn.model_selection import cross_val_predict

from customer_churn.features.prepare_features import (
    load_processed_data,
    prepare_features,
    split_data,
)

from customer_churn.training.tune_models import (
    build_cross_validator,
)

RANDOM_STATE = 42
MIN_RECALL = 0.75
DEFAULT_THRESHOLD = 0.50

MODEL_DIR = Path("artifacts/models")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

COMPARISON_FILE = REPORTS_DIR / "threshold_comparison.csv"
SUMMARY_FILE = REPORTS_DIR / "threshold_summary.json"

MODEL_FILES = {
    "Balanced Logistic Regression": (
        MODEL_DIR / "tuned_balanced_logistic_regression.joblib"
    ),
    "Gradient Boosting": (
        MODEL_DIR / "tuned_gradient_boosting.joblib"
    ),
}


def get_thresholds() -> list[float]:
    """Return threshold from 0.10 to 0.90"""

    return [
        round(0.10 + 0.05 * index, 2)
        for index in range(17)
    ]


def calculate_threshold_metrics (
        y_true,
        probabilities,
        threshold: float,
) -> dict:
    """Calculate classification metrics at one threshold"""

    predictions = [
        int(probability >= threshold)
        for probability in probabilities
    ]

    tn, fp, fn, tp = confusion_matrix (
        y_true,
        predictions,
        labels = [0, 1],
    ).ravel()

    return {
        "threshold": threshold,
        "accuracy": accuracy_score (
            y_true, predictions
        ),
        "precision": precision_score (
            y_true,
            predictions,
            zero_division = 0,
        ),
        "recall": recall_score (
            y_true,
            predictions,
            zero_division = 0,
        ),
        "f1": f1_score (
            y_true,
            predictions,
            zero_division = 0,
        ),
        "f2": fbeta_score (
            y_true,
            predictions,
            beta = 2,
            zero_division = 0,
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_threshold (
        results: pd.DataFrame,
        min_recall: float = MIN_RECALL,
) -> tuple[pd.Series, bool]:
    """Maximize precision while satisfying minimum recall"""

    if not 0 <= min_recall <= 1:
        raise ValueError (
            "min_recall must be between 0 and 1."
        )
    
    # Only consider threshold meeting the recall target
    eligible = results.loc [
        results["recall"] >= min_recall
    ]

    if not eligible.empty:
        selected = eligible.sort_values (
            by = ["precision", "f1", "threshold"],
            ascending = [False, False, False],
        ).iloc[0]

        return selected, True

    # Fallback if no threshold meets the recall target
    selected = results.sort_values (
        by = ["f2", "precision", "threshold"],
        ascending = [False, False, False],
    ).iloc[0]

    return selected, False


def format_metrics(row: pd.Series) -> dict:
    """Convert metric values into JSON-safe Python types"""

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f2",
    ]

    metrics = {
        name: float(row[name])
        for name in metric_names
    }

    for name in ["tn", "fp", "fn", "tp"]:
        metrics[name] = int(row[name])

    return metrics


def save_threshold_plot (
        results: pd.DataFrame,
        selected_threshold: float,
        filename: str,
) -> Path:
    """Visualize precision and recall across threshold"""

    FIGURES_DIR.mkdir (
        parents = True,
        exist_ok = True,
    )

    figure_path = FIGURES_DIR / filename

    fig, ax = plt.subplots(figsize = (9, 5))

    ax.plot (
        results["threshold"],
        results["precision"],
        marker = "o",
        label = "Precision",
    )

    ax.plot (
        results["threshold"],
        results["recall"],
        marker = "o",
        label = "Recall",
    )

    ax.plot (
        results["threshold"],
        results["f1"],
        marker = "o",
        label = "F1",
    )

    ax.axhline (
        MIN_RECALL,
        linestyle = ":",
        label = "Recall target",
    )

    ax.axvline (
        selected_threshold,
        linestyle = "--",
        label = "Selected threshold",
    )

    ax.set_title("Precision and Recall by Threshold")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend()

    fig.tight_layout()
    fig.savefig(figure_path, dpi = 150)

    plt.close(fig)

    return figure_path


def main() -> None:
    """Optimize thresholds using training-only OOF scores"""

    dataframe = load_processed_data()

    X, y = prepare_features(dataframe)

    X_train, _, y_train, _ = split_data(X, y)

    REPORTS_DIR.mkdir (
        parents = True,
        exist_ok = True,
    )

    summary = {}
    all_results = []

    print(f"Training records: {len(X_train)}")
    print("Test set excluded from threshold optimization.")

    for model_name, model_file in MODEL_FILES.items():

        print()
        print("=" * 50)
        print(f"Optimizing: {model_name}")
        print("=" * 50)

        if not model_file.exists():
            raise FileNotFoundError (
                f"Missing model artifact: {model_file}"
            )

        # Load only artifacts generated by the project
        model = joblib.load(model_file)

        # Each prediction comes from a model fitted without that record in its training fold
        oof_probabilities = cross_val_predict (
            estimator = model,
            X = X_train,
            y = y_train,
            cv = build_cross_validator(),
            method = "predict_proba",
            n_jobs = 1,
        )[:, 1]

        rows = [
            calculate_threshold_metrics (
                y_train,
                oof_probabilities,
                threshold,
            )
            for threshold in get_thresholds()
        ]

        results = pd.DataFrame(rows)

        selected, target_met = select_threshold (results)

        # Prevent an invalid threshold from being accepted
        if target_met and selected["recall"] < MIN_RECALL:
            raise ValueError (
                "Selected threshold violates the minimum recall requiremnet."
            )

        default = results [
            results["threshold"] == DEFAULT_THRESHOLD
        ].iloc[0]

        print()
        print("Default threshold:")
        print(f"Threshold: {DEFAULT_THRESHOLD:.2f}")
        print(f"Precision: {default['precision']:.4f}")
        print(f"Recall:    {default['recall']:.4f}")
        print(f"F1:        {default['f1']:.4f}")

        print()
        print(f"Selected threshold:")
        print(f"Threshold: {selected['threshold']:.2f}")
        print(f"Precision: {selected['precision']:.4f}")
        print(f"Recall:    {selected['recall']:.4f}")
        print(f"F1:        {selected['f1']:.4f}")

        print()
        print(f"Recall target met: {target_met}")

        summary[model_name] = {
            "selected_threshold": float (
                selected["threshold"]
            ),
            "recall_target": MIN_RECALL,
            "recall_target_met": bool(target_met),
            "default_metrics": format_metrics(default),
            "selected_metrics": format_metrics(selected),
            "selection_data": "training_only_oof",
        }

        results["model"] = model_name

        all_results.append(results)

        filename = (
            "threshold_"
            + model_file.stem
            + ".png"
        )

        save_threshold_plot (
            results,
            float(selected["threshold"]),
            filename,
        )

    comparison = pd.concat (
        all_results,
        ignore_index = True,
    )

    comparison.to_csv (
        COMPARISON_FILE,
        index = False,
    )

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
    print("Threshold optimization completed.")
    print(f"Comparison saved to: {COMPARISON_FILE}")
    print(f"Summary saved to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()