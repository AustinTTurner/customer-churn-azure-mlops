"""Tests for shared project configuration"""

import json
import pytest

from customer_churn.config import load_azure_config


def test_default_azure_config_loads() -> None:
    """The committed Azure configuration shout load successfully"""

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
        config["deployment"]["instance_type"]
        == "Standard_DS2_v2"
    )


def test_missing_config_file_raise (
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