"""Tests for candidate model comparison"""

from sklearn.pipeline import Pipeline

from customer_churn.evaluation.compare_models import (
    build_candidate_models,
    )


def test_expected_candidate_models_exists():
    models = build_candidate_models()

    expected_models = {
        "Logistic Regression",
        "Balanced Logistic Regression",
        "Random Forest",
        "Gradient Boosting",
    }

    assert set(models.keys()) == expected_models


def test_candidate_models_are_pipelines():
    models = build_candidate_models()

    for model in models.values():
        assert isinstance (
            model,
            Pipeline,
        )


def test_candidate_models_include_required_steps():
    models = build_candidate_models()

    for model in models.values():

        assert "preprocessor" in model.named_steps
        assert "classifier" in model.named_steps