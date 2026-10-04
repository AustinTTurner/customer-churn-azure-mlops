"""Register the cleaned churn dataset in Azure Machine Learning"""

import os
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError

from customer_churn.config import load_azure_config

LOCAL_DATA_PATH = Path (
    "data/processed/telco_customer_churn_clean.csv"
)


def main() -> None:
    """Upload and register the cleaned dataset"""

    config = load_azure_config()

    resource_group = config["azure"]["resource_group"]
    workspace_name = config["azure"]["workspace_name"]

    data_config = config["assets"]["data"]
    data_asset_name = data_config["name"]
    data_asset_version = data_config["version"]

    project_name = config["project"]["name"]

    subscription_id = os.environ.get (
        "AZURE_SUBSCRIPTION_ID"
    )

    if not subscription_id:
        raise RuntimeError (
            "AZURE_SUBSCRIPTION_ID is not set."
        )

    if not LOCAL_DATA_PATH.is_file():
        raise FileNotFoundError (
            f"Dataset not found: {LOCAL_DATA_PATH}"
        )

    credential = DefaultAzureCredential()

    ml_client = MLClient (
        credential = credential,
        subscription_id = subscription_id,
        resource_group_name = resource_group,
        workspace_name = workspace_name,
    )

    data_asset = Data (
        name = data_asset_name,
        version = data_asset_version,
        description = (
            "Cleaned IBM Telco Customer Churn dataset used for Azure ML training."
        ),
        path = str(LOCAL_DATA_PATH),
        type = AssetTypes.URI_FILE,
        tags = {
            "project": project_name,
            "stage": "cleaned",
        },
    )

    try:
        registered = ml_client.data.get (
            name = data_asset_name,
            version = data_asset_version,
        )

        print()
        print(
            "Azure ML data asset already exists. "
            "Reusing the registered version."
        )

    except ResourceNotFoundError:
        registered = ml_client.data.create_or_update (
            data_asset
        )
        
        print()
        print("Azure ML data asset registered successfully.")


    print(f"Name: {registered.name}")
    print(f"Version: {registered.version}")
    print(f"Type: {registered.type}")
    print(f"Path: {registered.path}")


if __name__ == "__main__":
    main()