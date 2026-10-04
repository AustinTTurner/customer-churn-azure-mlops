"""Shared project configuration utilities"""

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_AZURE_CONFIG_PATH = (
    REPO_ROOT / "configs" / "azure.json"
)

REQUIRED_SECTIONS = {
    "project",
    "azure",
    "assets",
    "experiments",
    "deployment",
    "tags",
}

REQUIRED_PROJECT_FIELDS = {
    "name",
    "environment",
}

REQUIRED_AZURE_FIELDS = {
    "location",
    "resource_group",
    "workspace_name",
}

REQUIRED_ASSET_FIELDS = {
    "data": {"name", "version"},
    "training_environment": {"name", "version"},
    "inference_environment": {"name", "version"},
    "model": {"name", "version"},
}

REQUIRED_EXPERIMENT_FIELDS = {
    "training",
    "pipeline",
}

REQUIRED_DEPLOYMENT_FIELDS = {
    "name",
    "instance_type",
    "instance_count",
}

REQUIRED_TAG_FIELDS = {
    "project",
    "environment",
}


def load_azure_config (
        path: str | Path | None = None,
) -> dict[str, Any]:
    """Load and validate the shared Azure project configuration"""

    config_path = (
        Path(path)
        if path is not None
        else DEFAULT_AZURE_CONFIG_PATH
    )

    if not config_path.is_file():
        raise FileNotFoundError (
            f"Azure configuration file not found: "
            f"{config_path}"
        )

    with config_path.open (
        "r",
        encoding = "utf-8",
    ) as file_handle:
        config = json.load(file_handle)

    missing_sections = sorted (
        REQUIRED_SECTIONS - set(config)
    )

    if missing_sections:
        raise ValueError (
            "Azure configuration is missing required "
            f"sections: {missing_sections}"
        )

    missing_project_fields = sorted (
        REQUIRED_PROJECT_FIELDS
        - set(config["project"])
    )

    if missing_project_fields:
        raise ValueError (
            "Azure configuration is missing required "
            f"project fields: {missing_project_fields}"
        )

    missing_azure_fields = sorted (
        REQUIRED_AZURE_FIELDS
        - set(config["azure"])
    )

    if missing_azure_fields:
        raise ValueError (
            "Azure configuration in missing required "
            f"Azure fields: {missing_azure_fields}"
        )

    for asset_name, required_fields in REQUIRED_ASSET_FIELDS.items():
        if asset_name not in config["assets"]:
            raise ValueError (
                "Azure configuration is missing required "
                f"asset section: {asset_name}"
            )

        missing_asset_fields = sorted (
            required_fields
            - set(config["assets"][asset_name])
        )

        if missing_asset_fields:
            raise ValueError (
                "Azure configuration assets "
                f"'{asset_name}' is missing required fields: "
                f"{missing_asset_fields}"
            )

        missing_experiment_fields = sorted (
            REQUIRED_EXPERIMENT_FIELDS
            - set(config["experiments"])
        )

        if missing_experiment_fields:
            raise ValueError (
                "Azure configuration is missing required "
                f"experiment fields: {missing_experiment_fields}"
            )

        missing_deployment_fields = sorted (
            REQUIRED_DEPLOYMENT_FIELDS
            - set(config["deployment"])
        )

        if missing_deployment_fields:
            raise ValueError (
                "Azure configuration is missing required "
                f"deployment fields: {missing_deployment_fields}"
            )

        missing_tag_fields = sorted (
            REQUIRED_TAG_FIELDS
            - set(config["tags"])
        )

        if missing_tag_fields:
            raise ValueError (
                "Azure configuration is missing required "
                f"tag fields: {missing_tag_fields}"
            )

    return config