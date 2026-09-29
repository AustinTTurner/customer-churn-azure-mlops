"""Tests for decision-threshold optimization"""

import pandas as pd
import pytest

from customer_churn.evaluation.optimize_threshold import (
    calculate_threshold_metrics,
    get_thresholds,
    select_threshold,
)


def sample_predictions():
    """Return controlled labels and prediction scores"""

    y_true = [1, 1, 0, 0]

    probabilities = [
        0.90,
        0.45,
        0.35,
        0.10,
    ]

    return y_true, probabilities


def test_default_threshold():
    y_true, probabilities = sample_predictions()

    metrics = calculate_threshold_metrics (
        y_true,
        probabilities,
        threshold = 0.50,
    )

    assert metrics["tp"] == 1
    assert metrics["fn"] == 1
    assert metrics["fp"] == 0
    assert metrics["tn"] == 2

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5


def test_lower_threshold_increases_recall():
    y_true, probabilities = sample_predictions()

    default = calculate_threshold_metrics (
        y_true, probabilities, 0.50
    )

    lower = calculate_threshold_metrics (
        y_true, probabilities, 0.40
    )

    assert lower["recall"] > default["recall"]


def test_selection_meets_recall_target():
    y_true, probabilities = sample_predictions()

    results = pd.DataFrame (
        [
            calculate_threshold_metrics (
                y_true, probabilities, threshold
            )
            for threshold in [0.30, 0.40, 0.50]
        ]
    )

    selected, target_met = select_threshold (
        results,
        min_recall = 0.75,
    )

    assert target_met is True
    assert selected["threshold"] == pytest.approx(0.40)
    assert selected["recall"] >= 0.75


def test_fallback_when_target_not_met():
    y_true, probabilities = sample_predictions()

    results = pd.DataFrame (
        [
            calculate_threshold_metrics (
                y_true, probabilities, 0.50
            )
        ]
    )

    selected, target_met = select_threshold (
        results,
        min_recall = 0.75,
    )

    assert target_met is False
    assert selected["threshold"] == pytest.approx(0.50)


def test_threshold_grid():
    thresholds = get_thresholds()

    assert len(thresholds) == 17
    assert thresholds[0] == 0.10
    assert thresholds[-1] == 0.90
    assert 0.50 in thresholds


def test_rejects_high_precision_low_recall():
    """Reject a high-precision threshold that misses recall"""

    y_true = [1, 1, 1, 1, 0, 0, 0, 0]

    probabilities = [
        0.95,
        0.70,
        0.65,
        0.45,
        0.80,
        0.60,
        0.20,
        0.10,
    ]

    results = pd.DataFrame (
        [
            calculate_threshold_metrics (
                y_true,
                probabilities,
                threshold,
            )
            for threshold in [0.40, 0.50, 0.90]
        ]
    )

    selected, target_met = select_threshold (
        results,
        min_recall = 0.75,
    )

    assert target_met is True
    assert selected["threshold"] == pytest.approx(0.40)
    assert selected["recall"] >= 0.75