"""Tests for hyperparameter tuning configuration"""

from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    StratifiedKFold,
)

from customer_churn.evaluation.compare_models import (
    build_candidate_models,
)

from customer_churn.training.tune_models import (
    build_cross_validator,
    build_search,
    get_tuning_configs,
)


def test_cross_validator_configuration():
    cv = build_cross_validator()

    assert isinstance(cv, StratifiedKFold)
    assert cv.n_splits == 5
    assert cv.shuffle is True
    assert cv.random_state == 42


def test_expected_tuning_candidates():
    configs = get_tuning_configs()

    assert set(configs) == {
        "Balanced Logistic Regression",
        "Gradient Boosting",
    }


def test_parameter_grid_sizes():
    configs = get_tuning_configs()

    logistic_grid = configs [
        "Balanced Logistic Regression"
    ]["param_grid"]

    boosting_grid = configs[
        "Gradient Boosting"
    ]["param_grid"]

    assert len(list(ParameterGrid(logistic_grid))) == 3
    assert len(list(ParameterGrid(boosting_grid))) == 8


def test_grid_search_configuration():
    models = build_candidate_models()
    configs = get_tuning_configs()

    for name, config in configs.items():
        search = build_search (
            models[name],
            config["param_grid"],
        )

        assert isinstance(search, GridSearchCV)
        assert search.refit == "f1"
        assert search.cv.n_splits == 5

        assert set(search.scoring) == {
            "precision",
            "recall",
            "f1",
            "roc_auc",
        }