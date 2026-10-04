"""Register the Azure ML cloud training environment"""

import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError

from customer_churn.config import load_azure_config


def main() -> None:
    """Register the cloud training environment"""

    config = load_azure_config()

    resource_group = config["azure"]["resource_group"]
    workspace_name = config["azure"]["workspace_name"]

    environment_config = config["assets"]["training_environment"]
    environment_name = environment_config["name"]
    environment_version = environment_config["version"]

    project_name = config["project"]["name"]

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
        resource_group_name = resource_group,
        workspace_name = workspace_name,
    )

    environment = Environment (
        name = environment_name,
        version = environment_version,
        description = (
            "Reproducible Azure ML environment for customer churn model training."
        ),
        image = (
            "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest"
        ),
        conda_file = "azure/environments/train_conda.yml",
        tags = {
            "project": project_name,
            "purpose": "cloud-training",
        },
    )

    try:
        registered = ml_client.environments.get (
            name = environment_name,
            version = environment_version,
        )

        print()
        print(
            "Azure ML training environment already exists. "
            "Reusing the registered version."
        )

    except ResourceNotFoundError:
        registered = (
            ml_client.environments.create_or_update (
                environment
            )
        )

        print()
        print(
            "Azure ML training environment registered successfully."
        )

    print(f"Name: {registered.name}")
    print(f"Version: {registered.version}")


if __name__ == "__main__":
    main()