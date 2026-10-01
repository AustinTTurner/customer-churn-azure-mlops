"""Register the cleaned churn dataset in Azure Machine Learning"""

import os
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential

RESOURCE_GROUP = "rg-customer-churn-mlops-dev"
WORKSPACE_NAME = "mlw-customer-churn-dev"

DATA_ASSET_NAME = "telco-customer-churn-clean"
DATA_ASSET_VERSION = "1"

LOCAL_DATA_PATH = Path (
    "data/processed/telco_customer_churn_clean.csv"
)


def main() -> None:
    """Upload and register the cleaned dataset"""

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
        resource_group_name = RESOURCE_GROUP,
        workspace_name = WORKSPACE_NAME,
    )

    data_asset = Data (
        name = DATA_ASSET_NAME,
        version = DATA_ASSET_VERSION,
        description = (
            "Cleaned IBM Telco Customer Churn dataset used for Azure ML training."
        ),
        path = str(LOCAL_DATA_PATH),
        type = AssetTypes.URI_FILE,
        tags = {
            "project": "customer-churn-azure-mlops",
            "stage": "cleaned",
        },
    )

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