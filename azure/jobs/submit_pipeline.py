"""Submit the customer churn Azure ML pipeline"""

import os
import time

from azure.ai.ml import (
    Input,
    MLClient,
    dsl,
    load_component,
)

from azure.ai.ml.entities import (
    UserIdentityConfiguration,
)

from azure.identity import DefaultAzureCredential

RESOURCE_GROUP = "rg-customer-churn-mlops-dev"
WORKSPACE_NAME = "mlw-customer-churn-dev"


validate_component = load_component (
    source = "azure/components/validate_data.yml"
)

train_component = load_component (
    source = "azure/components/train_model.yml"
)

package_component = load_component (
    source = "azure/components/package_model.yml"
)


@dsl.pipeline (
    name = "customer-churn-mlops-pipeline",
    description = (
        "Validate data, train the churn model, and package the deployment artifact."
    ),
)


def customer_churn_pipeline (
    pipeline_input,
):
    """Define the customer churn MLOps pipeline"""

    validate_job = validate_component (
        input_data = pipeline_input
    )

    validate_job.identity = (
        UserIdentityConfiguration()
    )

    train_job = train_component (
        validated_data = (
            validate_job.outputs.validated_data
        )
    )

    train_job.identity = (
        UserIdentityConfiguration()
    )

    package_job = package_component (
        model_input = (
            train_job.outputs.model_output
        )
    )

    package_job.identity = (
        UserIdentityConfiguration()
    )

    return {
        "validated_data": (
            validate_job.outputs.validated_data
        ),
        "trained_model": (
            train_job.outputs.model_output
        ),
        "packaged_model": (
            package_job.outputs.package_output
        ),
    }


def main() -> None:
    """Submit and monitor the Azure ML pipeline"""

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

    pipeline_job = customer_churn_pipeline (
        pipeline_input=Input (
            type = "uri_file",
            path = (
                "azureml:telco-customer-churn-clean:1"
            ),
            mode = "download",
        ),
    )

    pipeline_job.settings.default_compute = (
        "serverless"
    )

    pipeline_job.display_name = (
        "customer-churn-mlops-pipeline"
    )

    pipeline_job.tags = {
        "project": "customer-churn-azure-mlops",
        "stage": "pipeline",
    }

    returned_job = (
        ml_client.jobs.create_or_update (
            pipeline_job,
            experiment_name = (
                "customer-churn-mlops-pipeline"
            ),
        )
    )

    print()
    print("Azure ML pipeline submitted.")
    print(f"Pipeline job: {returned_job.name}")

    studio_service = (
        returned_job.services.get("Studio")
    )

    if studio_service:
        print(
            f"Studio URL: "
            f"{studio_service.endpoint}"
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
            f"Current pipeline status: "
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
        f"Final pipeline status: "
        f"{final_job.status}"
    )

    if final_job.status != "Completed":
        raise RuntimeError (
            "Azure ML pipeline did not complete successfully."
        )


if __name__ == "__main__":
    main()