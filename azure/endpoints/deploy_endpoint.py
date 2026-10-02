"""Deploy the churn model to an Azure ML managed endpoint"""

import argparse
import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    CodeConfiguration,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
)

from azure.identity import DefaultAzureCredential


RESOURCE_GROUP = "rg-customer-churn-mlops-dev"
WORKSPACE_NAME = "mlw-customer-churn-dev"

MODEL = (
    "azureml:"
    "customer-churn-balanced-logistic:2"
)

ENVIRONMENT = (
    "azureml:"
    "customer-churn-inference:2"
)


def main() -> None:
    """Create the endpoint and blue deployment"""

    parser = argparse.ArgumentParser()

    parser.add_argument (
        "--endpoint-name",
        required = True,
        help = ("Unique Azure ML managed online endpoint name."),
    )

    args = parser.parse_args()

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

    endpoint = ManagedOnlineEndpoint (
        name = args.endpoint_name,
        description = (
            "Real-time customer churn prediction endpoint."
        ),
        auth_mode = "key",
        tags = {
            "project": (
                "customer-churn-azure-mlops"
            ),
            "stage": "portfolio-demo"
        },
    )

    print()
    print(
        f"Creating endpoint: "
        f"{args.endpoint_name}"
    )

    ml_client.online_endpoints.begin_create_or_update (
        endpoint
    ).result()

    deployment = ManagedOnlineDeployment (
        name = "blue",
        endpoint_name = args.endpoint_name,
        model = MODEL,
        environment = ENVIRONMENT,
        code_configuration = CodeConfiguration (
            code = "azure/endpoints/scoring",
            scoring_script = "score.py",
        ),
        instance_type = "Standard_DS3_v2",
        instance_count = 1,
        app_insights_enabled = True,
    )

    print()
    print("Creating blue deployment...")

    ml_client.online_deployments.begin_create_or_update (
        deployment
    ).result()

    endpoint = (
        ml_client.online_endpoints.get (
            args.endpoint_name
        )
    )

    endpoint.traffic = {
        "blue": 100
    }

    ml_client.online_endpoints.begin_create_or_update (
        endpoint
    ).result()

    final_endpoint = (
        ml_client.online_endpoints.get (
            args.endpoint_name
        )
    )

    print()
    print(
        "Managed online endpoint deployment completed."
    )

    print(
        f"Endpoint: "
        f"{final_endpoint}"
    )

    print(
        f"Provisioning state: "
        f"{final_endpoint.provisioning_state}"
    )

    print(
        "Traffic: blue = 100%"
    )


if __name__ == "__main__":
    main()