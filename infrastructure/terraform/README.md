# Terraform Infrastructure

This directory contains the Terraform configuration used to provision the foundational Azure infrastructure for the Customer Churn Azure Machine Learning project.

Terraform is responsible for the cloud infrastructure layer. Azure Machine Learning assets and workloads are managed separately through the Azure ML SDK and Azure CLI.

This separation allows infrastructure provisioning and machine learning orchestration to evolve independently while still sharing project-level configuration.

---

## Infrastructure Ownership

Terraform manages the following Azure resources:

- Azure Resource Group
- Azure Storage Account
- Azure Key Vault
- Azure Log Analytics Workspace
- Azure Application Insights
- Azure Container Registry
- Azure Machine Learning Workspace

Terraform also manages a `random_id` resource used to create globally unique names for Azure services such as the Storage Account, Key Vault, and Container Registry.

Azure Machine Learning resources such as the following are intentionally not managed by Terraform:

- Azure ML data assets
- Azure ML environments
- Training jobs
- Azure ML pipelines
- Registered models
- Managed online endpoints
- Online deployments

Those resources are managed through the Azure Machine Learning SDK and Azure CLI.

The ownership boundary is therefore:

```text
                  Azure Environment
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
     Terraform                       Azure ML SDK
        │                                 │
        ├── Resource Group                ├── Data assets
        ├── Storage Account               ├── Environments
        ├── Key Vault                     ├── Jobs
        ├── Log Analytics                 ├── Pipelines
        ├── Application Insights          ├── Models
        ├── Container Registry            └── Endpoints
        └── Azure ML Workspace
```

---

## Shared Configuration

Terraform reads project-level Azure settings from:

```text
../../configs/azure.json
```

The same configuration file is also used by the project's Python-based Azure ML orchestration scripts.

The shared configuration includes values such as:

- Azure region
- Resource group name
- Azure ML workspace name
- Project name
- Environment name
- Azure ML asset names
- Asset versions
- Deployment configuration
- Project tags

This reduces duplicated configuration between the infrastructure and machine learning layers.

The Azure subscription ID is intentionally not stored in `configs/azure.json`.

For Terraform, the subscription ID is supplied through the AzureRM provider environment:

```powershell
$env:ARM_SUBSCRIPTION_ID = az account show --query id --output tsv
```

---

## Terraform Version

The configuration requires Terraform within the supported 1.x range.

The project was validated using:

```text
Terraform 1.16.4
```

The version constraint is defined in `versions.tf`.

The project currently uses:

```text
AzureRM Provider: ~> 5.8
Random Provider:  ~> 3.9
```

Provider versions are resolved and recorded in:

```text
.terraform.lock.hcl
```

The lock file is intentionally committed so future Terraform initialization uses a consistent provider dependency resolution.

---

## Directory Structure

```text
infrastructure/terraform/
│
├── .gitignore
├── .terraform.lock.hcl
├── README.md
├── locals.tf
├── main.tf
├── outputs.tf
├── providers.tf
├── terraform.tfvars.example
├── variables.tf
└── versions.tf
```

### File Responsibilities

`versions.tf`

Defines the required Terraform version and provider version constraints.

`providers.tf`

Configures the AzureRM provider and provider-specific behavior.

`variables.tf`

Defines Terraform-specific configurable values and validation rules.

`locals.tf`

Loads the shared project configuration from `configs/azure.json` and creates reusable local values.

`main.tf`

Defines the Azure resources managed by Terraform.

`outputs.tf`

Exposes useful non-secret infrastructure values after deployment.

`terraform.tfvars.example`

Documents optional Terraform-specific overrides without committing a real `.tfvars` file.

`.terraform.lock.hcl`

Records the selected provider versions.

---

## Provisioned Azure Resources

The Terraform configuration provisions the following foundational services.

### Resource Group

Provides the lifecycle boundary for the project's Azure development resources.

### Storage Account

Provides storage used by the Azure Machine Learning workspace.

The account is configured as a Standard StorageV2 account with locally redundant storage by default.

### Azure Key Vault

Provides the Key Vault dependency required by Azure Machine Learning.

### Log Analytics Workspace

Provides centralized Azure logging infrastructure.

The development configuration uses a 30-day retention period.

### Application Insights

Provides the Application Insights dependency associated with the Azure ML workspace.

Application Insights is linked to the Terraform-managed Log Analytics workspace.

### Azure Container Registry

Provides the container registry associated with the Azure Machine Learning workspace.

The development configuration uses the Basic SKU.

### Azure Machine Learning Workspace

Provides the cloud workspace used for:

- Data registration
- Environment registration
- Training jobs
- Azure ML pipelines
- Model registration
- Managed online endpoints

The workspace uses a system-assigned managed identity.

---

## Resource Naming

Several Azure resources require globally unique names.

Terraform uses:

```hcl
resource "random_id" "resource_suffix"
```

to generate a stable unique suffix for applicable resources.

The suffix is preserved in Terraform state for the lifetime of the infrastructure so resource names remain consistent between plans and applies.

---

## Common Tags

Shared project tags are loaded from:

```text
../../configs/azure.json
```

Terraform also adds:

```text
managed_by = terraform
```

to identify infrastructure owned by Terraform.

---

## Prerequisites

Before using this Terraform configuration, install:

- Terraform
- Azure CLI
- Git

You also need:

- An Azure subscription
- Permission to create the configured Azure resources
- An authenticated Azure CLI session

Authenticate with Azure:

```powershell
az login
```

Confirm the active subscription:

```powershell
az account show --output table
```

Expose the active subscription to the AzureRM provider:

```powershell
$env:ARM_SUBSCRIPTION_ID = az account show --query id --output tsv
```

Do not hardcode the subscription ID into the Terraform source files.

---

## Initialize Terraform

Run commands from the repository root.

Initialize the Terraform working directory:

```powershell
terraform -chdir=infrastructure/terraform init
```

Terraform will:

- Initialize the working directory
- Download required providers
- Read the dependency lock file
- Prepare the configuration for validation and planning

---

## Format

Terraform source files can be formatted with:

```powershell
terraform -chdir=infrastructure/terraform fmt
```

The `terraform.tfvars.example` file is an example configuration file and should be edited manually if needed.

---

## Validate

Validate the Terraform configuration:

```powershell
terraform -chdir=infrastructure/terraform validate
```

A successful validation returns:

```text
Success! The configuration is valid.
```

---

## Plan

Preview the infrastructure Terraform intends to create:

```powershell
terraform -chdir=infrastructure/terraform plan
```

Always review the plan before applying infrastructure changes.

For a saved plan:

```powershell
terraform -chdir=infrastructure/terraform plan -out=tfplan
```

Then apply the reviewed plan with:

```powershell
terraform -chdir=infrastructure/terraform apply tfplan
```

Saved Terraform plan files are excluded from source control.

---

## Apply

To provision the Azure infrastructure directly:

```powershell
terraform -chdir=infrastructure/terraform apply
```

Review the proposed changes before confirming the apply.

The completed development lifecycle successfully provisioned the foundational Azure infrastructure required by the Azure ML workflow.

---

## Verify the Azure ML Workspace

After Terraform finishes provisioning the infrastructure, the Python Azure ML layer can verify connectivity to the workspace.

The project includes:

```text
azure/sdk/verify_workspace.py
```

The Python Azure ML scripts use a separate environment variable:

```powershell
$env:AZURE_SUBSCRIPTION_ID = az account show --query id --output tsv
```

Terraform uses:

```text
ARM_SUBSCRIPTION_ID
```

while the project Python code uses:

```text
AZURE_SUBSCRIPTION_ID
```

Neither subscription ID is committed to source control.

---

## Azure ML Resource Ownership

After Terraform provisions the Azure ML workspace, the project uses Python and the Azure ML SDK to create and manage machine learning assets.

Examples include:

```text
azure/sdk/register_data.py
azure/sdk/register_environment.py
azure/sdk/register_inference_environment.py
azure/sdk/register_model.py

azure/jobs/submit_training_job.py
azure/jobs/submit_pipeline.py

azure/endpoints/deploy_endpoint.py
```

This means deleting or changing an Azure ML model, job, data asset, or endpoint does not normally represent Terraform drift because those resources are intentionally outside Terraform ownership.

That boundary was demonstrated during the completed project lifecycle: the managed online endpoint was deleted, and Terraform still reported zero infrastructure drift.

---

## Drift Detection

Terraform can compare the deployed Azure infrastructure with the committed configuration.

Run:

```powershell
terraform -chdir=infrastructure/terraform plan -detailed-exitcode
```

Terraform uses the following exit codes:

```text
0 = Succeeded and no infrastructure differences were detected
1 = Terraform encountered an error
2 = Succeeded and infrastructure changes were detected
```

During the completed project lifecycle, Terraform returned:

```text
No changes. Your infrastructure matches the configuration.
```

followed by:

```text
0
```

This verified that the Terraform-managed infrastructure matched the committed configuration.

Evidence is available at:

```text
../../docs/screenshots/terraform/zero-drift.png
```

---

## Destroy

The Azure environment used by this project is intended to be temporary development infrastructure.

A destroy plan can be created with:

```powershell
terraform -chdir=infrastructure/terraform plan `
    -destroy `
    -out=tfdestroyplan
```

Review the plan carefully.

The completed project lifecycle produced:

```text
Plan: 0 to add, 0 to change, 8 to destroy.
```

The reviewed plan can then be applied with:

```powershell
terraform -chdir=infrastructure/terraform apply tfdestroyplan
```

The completed teardown returned:

```text
Apply complete! Resources: 0 added, 0 changed, 8 destroyed.
```

The eight Terraform-managed objects included:

```text
random_id.resource_suffix
azurerm_resource_group.main
azurerm_storage_account.main
azurerm_key_vault.main
azurerm_log_analytics_workspace.main
azurerm_application_insights.main
azurerm_container_registry.main
azurerm_machine_learning_workspace.main
```

---

## Teardown Verification

After Terraform completed the destroy operation, Azure CLI was used to verify resource deletion.

Resource group existence was checked with:

```powershell
az group exists `
    --name rg-customer-churn-mlops-dev
```

Azure returned:

```text
false
```

A resource query subsequently returned:

```text
ResourceGroupNotFound
```

Terraform state was also checked:

```powershell
terraform -chdir=infrastructure/terraform state list
```

No managed resources remained.

Evidence of the successful teardown is available at:

```text
../../docs/screenshots/terraform/destroy.png
```

---

## Azure-Generated Supporting Resources

Some Azure services may automatically create supporting resources that are not explicitly represented as Terraform resources.

For example, Application Insights may create supporting monitoring resources.

Because this project uses a dedicated disposable development resource group, the AzureRM provider is configured to allow that resource group to be removed cleanly during Terraform teardown even when Azure has created supporting resources outside Terraform state.

This behavior is configured in:

```text
providers.tf
```

and is appropriate for this project's temporary development resource group.

A production environment should evaluate resource-group deletion behavior more conservatively.

---

## Terraform Variables

Terraform-specific values are defined in:

```text
variables.tf
```

Optional override examples are documented in:

```text
terraform.tfvars.example
```

The example file can be copied locally if overrides are required.

For example:

```powershell
Copy-Item `
    infrastructure/terraform/terraform.tfvars.example `
    infrastructure/terraform/terraform.tfvars
```

The real `terraform.tfvars` file is intentionally excluded from Git.

Do not store credentials or secrets in `.tfvars` files.

---

## Terraform Outputs

Terraform exposes non-secret infrastructure information through `outputs.tf`.

Examples include:

- Resource group name
- Azure ML workspace name
- Storage account name
- Key Vault name
- Log Analytics workspace name
- Application Insights name
- Container Registry name
- Container Registry login server
- Azure ML workspace resource ID

View outputs with:

```powershell
terraform -chdir=infrastructure/terraform output
```

Do not treat Terraform output as a mechanism for exposing credentials.

---

## State Management

This project uses local Terraform state because it is a single-developer development and demonstration environment.

Terraform state files are intentionally excluded from Git because state can contain sensitive infrastructure metadata.

Ignored Terraform runtime artifacts include:

```text
.terraform/
terraform.tfstate
terraform.tfstate.*
*.tfplan
tfplan
tfdestroyplan
```

These files should never be committed to the public repository.

### Production Recommendation

A collaborative or production environment should use a secured remote Terraform backend instead of local state.

For Azure, a common approach would be an Azure Storage backend with:

- Controlled access
- Encryption
- State locking
- Versioning
- Backup and recovery procedures
- Separate state per environment

Remote state is intentionally left as a future improvement rather than being represented as functionality already implemented by this project.

---

## Git Safety Checks

Before committing infrastructure changes, verify that Terraform runtime files are not tracked:

```powershell
git ls-files |
    Select-String "tfstate|tfplan|tfdestroyplan|\.terraform/"
```

A clean repository should return no Terraform runtime artifacts.

The dependency lock file should remain tracked:

```text
.terraform.lock.hcl
```

---

## Continuous Integration

Terraform configuration is automatically validated with GitHub Actions.

The CI workflow is defined in:
```text
../../.github/workflows/ci.yml
```

The Terraform validation job runs on pull requests targeting `main`, pushes to `main`, and manual workflow dispatch.

The workflow performs:
```text
terraform fmt -check -recursive
        │
        ▼
terraform init -backend=false -input=false
        │
        ▼
terraform validate -no-color
```

The workflow intentionally uses:
```
-backend=false
```

during initialization because CI only validates the Terraform configuration. It does not access or modify Terraform state.

The CI job also does not:

- Authenticate to Azure
- Run `terraform plan`
- Run `terraform apply`
- Run `terraform destroy`
- Create Azure resources
- Modify Azure infrastructure

This allows infrastructure configuration to be validated safely during pull-request review without requiring Azure credentials or generating cloud costs.

The Terraform CI job was successfully executed as part of the project's GitHub pull request workflow.

Continuous Deployment of Terraform infrastructure is not currently implemented.

---

## Infrastructure Lifecycle Demonstrated

The complete infrastructure lifecycle demonstrated by the project was:

```text
terraform init
      │
      ▼
terraform validate
      │
      ▼
terraform plan
      │
      ▼
terraform apply
      │
      ▼
Azure ML workspace verification
      │
      ▼
Azure ML data/environment registration
      │
      ▼
Cloud training
      │
      ▼
Azure ML pipeline
      │
      ▼
Model registration
      │
      ▼
Managed online endpoint
      │
      ▼
Live inference
      │
      ▼
Online endpoint deletion
      │
      ▼
terraform plan -detailed-exitcode
      │
      ▼
Zero infrastructure drift
      │
      ▼
terraform plan -destroy
      │
      ▼
terraform apply destroy plan
      │
      ▼
Azure resource group deletion verified
      │
      ▼
Terraform state empty
```

This validates both creation and controlled teardown of the project's development infrastructure.

---

## Evidence

Terraform lifecycle screenshots are stored in:

```text
../../docs/screenshots/terraform/
```

Current evidence includes:

```text
plan.png
zero-drift.png
destroy.png
```

These screenshots demonstrate:

- Infrastructure planning before deployment
- Zero detected Terraform drift after the Azure ML workload
- Successful Terraform-controlled teardown

---

## Security Considerations

The Terraform configuration is designed for a development environment rather than a hardened production platform.

The project intentionally avoids committing:

- Azure subscription IDs in Terraform source
- Credentials
- Terraform state
- Saved plan files
- Real `.tfvars` files

Production infrastructure would require additional security controls depending on organizational requirements.

---

## Production Improvements

Potential production-oriented infrastructure improvements include:

- Azure Storage remote Terraform backend
- State locking
- Separate state per environment
- Separate development, staging, and production environments
- Private networking
- Azure Private Link
- Restricted public network access
- Stronger Azure RBAC
- Managed identities for automation
- More restrictive Key Vault access
- Environment-controlled Continuous Deployment with approval gates
- Policy enforcement
- Security scanning
- Cost controls
- Centralized monitoring and alerting

These items are documented as future improvements and are not represented as functionality already implemented by the current project.

---

## Related Project Documentation

For the complete machine learning and MLOps workflow, including model development, Azure ML pipelines, deployment, inference, testing, evaluation methodology, and project results, see:

```text
../../README.md
```