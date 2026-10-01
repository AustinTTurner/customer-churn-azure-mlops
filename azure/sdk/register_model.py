"""Register the cloud-trained churn model in Azure Machine Learning"""

import argparse
import os

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

RESOURCE_GROUP = "rg-customer-churn-mlops-dev"
WORKSPACE_NAME = "mlw-customer-churn-dev"

MODEL_NAME = "customer-churn-balanced-logistic"


def main() -> None:
    """Register a completed Azure ML job's model output"""

    parser = argparse.ArgumentParser()

    parser.add_argument (
        "--job-name",
        required = True,
        help = "Azure ML training job name.",
    )

    parser.add_argument (
        "--model-version",
        default = "1",
        help = "Azure ML model asset version.",
    )

    parser.add_argument (
        "--output-name",
        default = "model_output",
        help = "Named Azure ML job output containing the model."
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

    job = ml_client.jobs.get(args.job_name)

    if job.status != "Completed":
        raise RuntimeError (
            f"Job {args.job_name} is not completed."
        )

    model_path = (
        f"azureml://jobs/{args.job_name}/outputs/{args.output_name}"
    )

    model = Model (
        name = MODEL_NAME,
        version = args.model_version,
        path = model_path,
        type = AssetTypes.CUSTOM_MODEL,
        description = (
            "Balanced logistic regression customer churn model trained in Azure Machine Learning."
        ),
        tags = {
            "project": "customer-churn-azure-mlops",
            "model_type": "balanced-logistic-regression",
            "decision_threshold": "0.55",
            "source_job": args.job_name,
            "source_output": args.output_name,
        },
    )

    registered = ml_client.models.create_or_update (
        model
    )

    print()
    print(f"Azure ML model registered successfully.")
    print(f"Name: {registered.name}")
    print(f"Version: {registered.version}")
    print(f"Type: {registered.type}")
    print(f"Source job: {args.job_name}")


if __name__ == "__main__":
    main()
