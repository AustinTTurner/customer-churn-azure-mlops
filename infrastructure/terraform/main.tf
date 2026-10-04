data "azurerm_client_config" "current" {}

resource "random_id" "resource_suffix" {
  byte_length = 4

  keepers = {
    resource_group_name = local.resource_group_name
  }
}

resource "azurerm_resource_group" "main" {
  name     = local.resource_group_name
  location = local.location

  tags = local.common_tags
}

resource "azurerm_storage_account" "main" {
  name = (
    "mlchurnst${random_id.resource_suffix.hex}"
  )

  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  account_tier             = "Standard"
  account_replication_type = var.storage_account_replication_type

  account_kind = "StorageV2"

  min_tls_version = "TLS1_2"

  allow_nested_items_to_be_public = false

  tags = local.common_tags
}

resource "azurerm_key_vault" "main" {
  name = (
    "mlchurn-kv-${random_id.resource_suffix.hex}"
  )

  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  tenant_id = data.azurerm_client_config.current.tenant_id
  sku_name  = "standard"

  rbac_authorization_enabled = false

  soft_delete_retention_days = (
    var.key_vault_soft_delete_retention_days
  )

  purge_protection_enabled = false

  tags = local.common_tags
}

resource "azurerm_log_analytics_workspace" "main" {
  name = (
    "log-customer-churn-${local.environment}"
  )

  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  sku = "PerGB2018"

  retention_in_days = (
    var.log_analytics_retention_days
  )

  tags = local.common_tags
}

resource "azurerm_application_insights" "main" {
  name = (
    "appi-customer-churn-${local.environment}"
  )

  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  application_type = "web"

  workspace_id = (
    azurerm_log_analytics_workspace.main.id
  )

  tags = local.common_tags
}

resource "azurerm_container_registry" "main" {
  name = (
    "mlchurnacr${random_id.resource_suffix.hex}"
  )

  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  sku           = var.container_registry_sku
  admin_enabled = true

  tags = local.common_tags
}

resource "azurerm_machine_learning_workspace" "main" {
  name = local.ml_workspace_name

  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  storage_account_id = (
    azurerm_storage_account.main.id
  )

  key_vault_id = (
    azurerm_key_vault.main.id
  )

  application_insights_id = (
    azurerm_application_insights.main.id
  )

  container_registry_id = (
    azurerm_container_registry.main.id
  )

  public_network_access_enabled = true

  identity {
    type = "SystemAssigned"
  }

  tags = local.common_tags
}