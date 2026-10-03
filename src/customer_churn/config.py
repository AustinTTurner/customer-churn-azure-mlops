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

    return config