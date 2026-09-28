"""Evaluate the baseline customer churn model"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from customer_churn.training.train_model import (
    train_model,
)

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

METRICS_FILE = REPORTS_DIR / "baseline_metrics.json"
CONFUSION_MATRIX_FILE = (
    FIGURES_DIR / "baseline_confusion_matrix.png"
)


def calculate_metrics (
        model,
        X_test,
        y_test,
) -> dict[str, float]:
    """Calculate classification metrics"""

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
        ),
        "recall": recall_score(
            y_test,
            predictions,
        ),
        "f1": f1_score(
            y_test,
            predictions,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    return metrics


def save_metrics(
        metrics: dict[str, float],
        path: Path = METRICS_FILE,
) -> Path:
    """Save evaluation metrics as JSON"""

    REPORTS_DIR.mkdir(
        parents = True,
        exist_ok = True,
    )

    with path.open(
        "w",
        encoding = "utf=8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent = 4,
        )

    return path


def save_confusion_matrix(
        model,
        X_test,
        y_test,
        path: Path = CONFUSION_MATRIX_FILE,
) -> Path:
    """Create and save confusion matrix"""

    FIGURES_DIR.mkdir(
        parents = True,
        exist_ok = True,
    )

    display = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels = [
            "No Churn",
            "Churn",
        ],
    )

    display.ax_.set_title(
        "Baseline Logistic Regression Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi = 150,
    )

    plt.close()

    return path


def main() -> None:
    """Train and evaluate the baseline model"""

    (
        model,
        _,
        X_test,
        _,
        y_test,
    ) = train_model()

    predictions = model.predict(
        X_test
    )

    metrics = calculate_metrics(
        model,
        X_test,
        y_test,
    )

    print()
    print("Baseline Model Evaluation")
    print("=" * 30)

    for metric_name, value in metrics.items():
        print(
            f"{metric_name:10}: "
            f"{value:.4f}"
        )

    print()
    print("Classification Report")
    print("=" * 30)

    print(
        classification_report(
            y_test,
            predictions,
            target_names = [
                "No Churn",
                "Churn",
            ],
        )
    )

    metrics_path = save_metrics(
        metrics
    )

    matrix_path = save_confusion_matrix(
        model,
        X_test,
        y_test,
    )

    print(
        f"Metrics saved to: "
        f"{metrics_path}"
    )

    print(
        f"Confusion matrix saved to: "
        f"{matrix_path}"
    )


if __name__ == "__main__":
    main()