"""Submit the customer churn training job to Azure ML"""

import os

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

RESOURCE_GROUP = "rg-customer-churn-mlops-dev"
WORKSPACE_NAME = "mlw-customer-churn-dev"


def main() -> None:
    """Submit and monitor the Azure ML training job"""

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
        workspace_name = WORKSPACE_NAME,
    )

    training_job = command (
        code = "src",
        command = (
            "python -m customer_churn.training.train_cloud --data ${{inputs.training_data}} --model-output ${{outputs.model_output}}"
        ),
        inputs = {
            "training_data": Input (
                type = AssetTypes.URI_FILE,
                path = "azureml:telco-customer-churn-clean:1",
                mode = "download",
            ),
        },
        outputs = {
            "model_output": Output (
                type = AssetTypes.URI_FOLDER,
                mode = "upload",
            ),
        },
        environment = "azureml:customer-churn-training:1",
        identity = UserIdentityConfiguration(),
        experiment_name = (
            "customer-churn-cloud-training"
        ),
        display_name = (
            "customer-churn-balanced-logistic"
        ),
        description = (
            "Train the selected customer churn classifier using Azure ML serverless compute."
        ),
        tags = {
            "project": "customer-churn-azure-mlops",
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

    ml_client.jobs.stream (
        returned_job.name
    )

    final_job = ml_client.jobs.get (
        returned_job.name
    )

    print()
    print(
        "Final job status: "
        f"{final_job.status}"
    )

    if final_job.status != "Completed":
        raise RuntimeError (
            "Azure ML training job did not complete."
        )


if __name__ == "__main__":
    main()