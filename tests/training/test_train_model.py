"""Tests for model training"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from customer_churn.training.train_model import (
    build_model_pipeline,
)


def test_model_pipeline_structure():
    model = build_model_pipeline()

    assert isinstance(
        model,
        Pipeline,
    )

    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps

    assert isinstance(
        model.named_steps["classifier"],
        LogisticRegression,
    )