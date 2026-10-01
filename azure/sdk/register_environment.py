"""Register the Azure ML cloud training environment"""

import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

RESOURCE_GROUP = "rg-customer-churn-mlops-dev"
WORKSPACE_NAME = "mlw-customer-churn-dev"

ENVIRONMENT_NAME = "customer-churn-training"
ENVIRONMENT_VERSION = "1"


def main() -> None:
    """Register the cloud training environment"""

    subscription_id = os.environ.get (
        "AZURE_SUBSCRIPTION_ID"
    )

    if not subscription_id:
        raise RuntimeError (
            "AZURE_SUBSCRIPTION_ID is not set."
        )

    ml_client = MLClient (
        credential = DefaultAzureCredential(),
        subscription_id = subscription_id,
        resource_group_name = RESOURCE_GROUP,
        workspace_name = WORKSPACE_NAME,
    )

    environment = Environment (
        name = ENVIRONMENT_NAME,
        version = ENVIRONMENT_VERSION,
        description = (
            "Reproducible Azure ML environment for customer churn model training."
        ),
        image = (
            "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest"
        ),
        conda_file = "azure/environments/train_conda.yml",
        tags = {
            "project": "customer-churn-azure-mlops",
            "purpose": "cloud-training",
        },
    )

    registered = (
        ml_client.environments.create_or_update (
            environment
        )
    )

    print()
    print("Azure ML environment registered successfully.")
    print(f"Name: {registered.name}")
    print(f"Version: {registered.version}")


if __name__ == "__main__":
    main()