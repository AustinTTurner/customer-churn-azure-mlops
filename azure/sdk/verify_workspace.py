"""Verify Azure Machine Learning workspace connectivity"""

import os

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from customer_churn.config import load_azure_config


def main() -> None:
    """Connect to Azure ML and verify workspace access"""

    config = load_azure_config()

    resource_group = config["azure"]["resource_group"]
    workspace_name = config["azure"]["workspace_name"]

    subscription_id = os.environ.get (
        "AZURE_SUBSCRIPTION_ID"
    )

    if not subscription_id:
        raise RuntimeError (
            "AZURE_SUBSCRIPTION_ID is not set."
        )

    credential = DefaultAzureCredential()

    ml_client = MLClient (
        credential = credential,
        subscription_id = subscription_id,
        resource_group_name = resource_group,
        workspace_name = workspace_name,
    )

    workspace = ml_client.workspaces.get (
        workspace_name
    )

    print()
    print("Azure ML SDK connection successful.")
    print(f"Workspace: {workspace.name}")
    print(f"Location: {workspace.location}")
    print(
        f"Resource Group: "
        f"{workspace.resource_group}"
    )

    computes = list (
        ml_client.compute.list()
    )

    print(
        f"Azure ML compute resources: "
        f"{len(computes)}"
    )


if __name__ == "__main__":
    main()