"""Tests for shared project configuration"""

import json
import pytest

from customer_churn.config import load_azure_config


def test_default_azure_config_loads() -> None:
    """The committed Azure configuration should load successfully"""

    config = load_azure_config()

    assert (
        config["azure"]["resource_group"]
        == "rg-customer-churn-mlops-dev"
    )

    assert (
        config["azure"]["workspace_name"]
        == "mlw-customer-churn-dev"
    )

    assert (
        config["assets"]["data"]["version"]
        == "1"
    )

    assert (
        config["assets"]["training_environment"]["version"]
        == "1"
    )

    assert (
        config["deployment"]["instance_type"]
        == "Standard_DS2_v2"
    )


def test_missing_config_file_raises (
        tmp_path,
) -> None:
    """A missing configuration file should fail clearly"""

    missing_path = (
        tmp_path / "missing.json"
    )

    with pytest.raises(FileNotFoundError):
        load_azure_config (
            missing_path
        )


def test_missing_required_section_raises (
        tmp_path,
) -> None:
    """Incomplete configuration should be rejected"""

    config_path = (
        tmp_path / "azure.json"
    )

    config_path.write_text (
        json.dumps (
            {
                "project": {},
                "azure": {},
            }
        ),
        encoding = "utf-8",
    )

    with pytest.raises (
        ValueError,
        match = "missing required sections",
    ):
        load_azure_config (
            config_path
        )


def test_missing_asset_version_raises (
        tmp_path,
) -> None:
    """Incomplete asset configuration should be rejected"""

    config_path = (
        tmp_path / "azure.json"
    )

    config_path.write_text (
        json.dumps (
            {
                "project": {
                    "name": "test-project",
                    "environment": "development",
                },
                "azure": {
                    "location": "southcentralus",
                    "resource_group": "test-rg",
                    "workspace_name": "test-workspace",
                },
                "assets": {
                    "data": {
                        "name": "test-data",
                    },
                    "training_environment": {
                        "name": "test-training",
                        "version": "1",
                    },
                    "inference_environment": {
                        "name": "test-inference",
                        "version": "1",
                    },
                    "model": {
                        "name": "test-model",
                        "version": "1",
                    },
                },
                "experiments": {
                    "training": "test-training-experiment",
                    "pipeline": "test-pipeline-experiment",
                },
                "deployment": {
                    "name": "blue",
                    "instance_type": "Standard_DS2_v2",
                    "instance_count": 1,
                },
                "tags": {
                    "project": "test-project",
                    "environment": "development",
                },
            }
        ),
        encoding = "utf-8",
    )

    with pytest.raises (
        ValueError,
        match = "data.*version",
    ):
        load_azure_config(
            config_path
        )