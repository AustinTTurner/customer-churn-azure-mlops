"""Submit the customer churn training job to Azure ML"""

import os
import time

from azure.ai.ml import (
    Input,
    MLClient,
    Output,
    command,
)
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    UserIdentityConfiguration,
)
from azure.identity import DefaultAzureCredential

from customer_churn.config import load_azure_config


def main() -> None:
    """Submit and monitor the Azure ML training job"""

    config = load_azure_config()

    resource_group = config["azure"]["resource_group"]
    workspace_name = config["azure"]["workspace_name"]

    data_config = config["assets"]["data"]
    training_environment_config = (
        config["assets"]["training_environment"]
    )

    experiment_name = config["experiments"]["training"]
    project_name = config["project"]["name"]

    data_reference = (
        f"azureml:{data_config['name']}:{data_config['version']}"
    )

    environment_reference = (
        f"azureml:{training_environment_config['name']}:"
        f"{training_environment_config['version']}"
    )

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

    training_job = command (
        code = "src",
        command = (
            "python -m customer_churn.training.train_cloud "
            "--data ${{inputs.training_data}} "
            "--model-output ${{outputs.model_output}}"
        ),
        inputs = {
            "training_data": Input (
                type = AssetTypes.URI_FILE,
                path = data_reference,
                mode = "download",
            ),
        },
        outputs = {
            "model_output": Output (
                type = AssetTypes.URI_FOLDER,
                mode = "upload",
            ),
        },
        environment = environment_reference,
        identity = UserIdentityConfiguration(),
        experiment_name = experiment_name,
        display_name = (
            "customer-churn-balanced-logistic"
        ),
        description = (
            "Train the selected customer churn classifier using Azure ML serverless compute."
        ),
        tags = {
            "project": project_name,
            "stage": "cloud-training",
            "model": "balanced-logistic-regression",
        },
    )

    returned_job = (
        ml_client.jobs.create_or_update(
            training_job
        )
    )

    print()
    print("Azure ML job submitted.")
    print(f"Job name: {returned_job.name}")

    studio_service = returned_job.services.get (
        "Studio"
    )

    if studio_service:
        print(
            f"Studio URL: "
            f"{studio_service.endpoint}"
        )

    print()
    print("Streaming Azure ML job logs...")
    print()

    try:
        ml_client.jobs.stream (
            returned_job.name
        )

    except Exception as exc:
        print()
        print(
            "Live log streaming was unavailable."
        )
        print(
            "Falling back to Azure ML status polling."
        )
        print(
            f"Streaming error: "
            f"{type(exc).__name__}: {exc}"
        )

        terminal_states = {
            "Completed",
            "Failed",
            "Canceled",
        }

        while True:
            current_job = ml_client.jobs.get (
                returned_job.name
            )

            print(
                f"Current job status: "
                f"{current_job.status}"
            )

            if current_job.status in terminal_states:
                break

            time.sleep(20)

    final_job = ml_client.jobs.get (
        returned_job.name
    )

    print()
    print(
        f"Final job status: "
        f"{final_job.status}"
    )

    if final_job.status != "Completed":
        raise RuntimeError (
            "Azure ML training job did not complete."
        )


if __name__ == "__main__":
    main()