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

from customer_churn.config import load_azure_config


def main() -> None:
    """Create the endpoint and managed online deployment"""

    config = load_azure_config()

    resource_group = config["azure"]["resource_group"]
    workspace_name = config["azure"]["workspace_name"]

    model_config = config["assets"]["model"]
    inference_environment_config = (
        config["assets"]["inference_environment"]
    )
    deployment_config = config["deployment"]

    project_name = config["project"]["name"]

    model_reference = (
        f"azureml:{model_config['name']}:"
        f"{model_config['version']}"
    )

    environment_reference = (
        f"azureml:{inference_environment_config['name']}:"
        f"{inference_environment_config['version']}"
    )

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
        resource_group_name = resource_group,
        workspace_name = workspace_name,
    )

    endpoint = ManagedOnlineEndpoint (
        name = args.endpoint_name,
        description = (
            "Real-time customer churn prediction endpoint."
        ),
        auth_mode = "key",
        tags = {
            "project": project_name,
            "stage": "online-deployment",
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
        name = deployment_config["name"],
        endpoint_name = args.endpoint_name,
        model = model_reference,
        environment = environment_reference,
        code_configuration = CodeConfiguration (
            code = "azure/endpoints/scoring",
            scoring_script = "score.py",
        ),
        instance_type = deployment_config["instance_type"],
        instance_count = deployment_config["instance_count"],
        app_insights_enabled = True,
    )

    print()
    print(
        f"Creating {deployment_config['name']} deployment..."
    )

    ml_client.online_deployments.begin_create_or_update (
        deployment
    ).result()

    endpoint = (
        ml_client.online_endpoints.get (
            args.endpoint_name
        )
    )

    endpoint.traffic = {
        deployment_config["name"]: 100
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
        f"{final_endpoint.name}"
    )

    print(
        f"Provisioning state: "
        f"{final_endpoint.provisioning_state}"
    )

    print(
        f"Traffic: "
        f"{deployment_config['name']} = 100%"
    )


if __name__ == "__main__":
    main()