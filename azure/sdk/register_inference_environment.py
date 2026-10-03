"""Register the Azure ML inference environment"""

import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential


RESOURCE_GROUP = "rg-customer-churn-mlops-dev"
WORKSPACE_NAME = "mlw-customer-churn-dev"


def main() -> None:
    """Register the versioned inference environment"""

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
        name = "customer-churn-inference",
        version = "2",
        description = (
            "Inference environment for the customer churn managed online endpoint."
        ),
        image = (
            "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest"
        ),
        conda_file = (
            "azure/environments/inference_conda.yml"
        ),
        tags = {
            "project": (
                "customer-churn-azure-mlops"
            ),
            "stage": "inference",
        },
    )

    registered_environment = (
        ml_client.environments.create_or_update (
            environment
        )
    )

    print()
    print(
        "Azure ML inference environment registered successfully."
    )

    print(
        f"Name: "
        f"{registered_environment.name}"
    )

    print(
        f"Version: "
        f"{registered_environment.version}"
    )


if __name__ == "__main__":
    main()
