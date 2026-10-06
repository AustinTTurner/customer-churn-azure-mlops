locals {
  project_config = jsondecode(
    file("${path.module}/../../configs/azure.json")
  )

  project_name = local.project_config.project.name
  environment  = local.project_config.project.environment

  location            = local.project_config.azure.location
  resource_group_name = local.project_config.azure.resource_group
  ml_workspace_name   = local.project_config.azure.workspace_name

  common_tags = merge(
    local.project_config.tags,
    {
      managed_by = "terraform"
    }
  )
}