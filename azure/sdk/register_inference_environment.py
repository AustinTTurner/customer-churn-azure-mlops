"""Register the Azure ML inference environment"""

import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError

from customer_churn.config import load_azure_config


def main() -> None:
    """Register the versioned inference environment"""

    config = load_azure_config()

    resource_group = config["azure"]["resource_group"]
    workspace_name = config["azure"]["workspace_name"]

    environment_config = config["assets"]["inference_environment"]
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
            "Inference environment for the customer churn managed online endpoint."
        ),
        image = (
            "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest"
        ),
        conda_file = (
            "azure/environments/inference_conda.yml"
        ),
        tags = {
            "project": project_name,
            "stage": "inference",
        },
    )

    try:
        registered_environment = (
            ml_client.environments.get (
                name = environment_name,
                version = environment_version,
            )
        )

        print()
        print(
            "Azure ML inference environment already exists. "
            "Reusing the registered version."
        )

    except ResourceNotFoundError:
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
