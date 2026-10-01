"""Verify Azure Machine Learning workspace connectivity"""

import os

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

RESOURCE_GROUP = "rg-customer-churn-mlops-dev"
WORKSPACE_NAME = "mlw-customer-churn-dev"


def main() -> None:
    """Connect to Azure ML and verify workspace access"""

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
        resource_group_name = RESOURCE_GROUP,
        workspace_name =WORKSPACE_NAME,
    )

    workspace = ml_client.workspaces.get (
        WORKSPACE_NAME
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