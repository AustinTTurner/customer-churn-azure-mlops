output "resource_group_name" {
  description = "Name of the Azure resource group."
  value       = azurerm_resource_group.main.name
}

output "machine_learning_workspace_name" {
  description = "Name of the Azure Machine Learning workspace."
  value       = azurerm_machine_learning_workspace.main.name
}

output "machine_learning_workspace_id" {
  description = "Resource ID of the Azure Machine Learning workspace."
  value       = azurerm_machine_learning_workspace.main.id
}

output "storage_account_name" {
  description = "Name of the Azure Storage Account used by Azure Machine Learning."
  value       = azurerm_storage_account.main.name
}

output "key_vault_name" {
  description = "Name of the Azure Key Vault used by Azure Machine Learning."
  value       = azurerm_key_vault.main.name
}

output "log_analytics_workspace_name" {
  description = "Name of the Log Analytics workspace."
  value       = azurerm_log_analytics_workspace.main.name
}

output "application_insights_name" {
  description = "Name of the Application Insights resource."
  value       = azurerm_application_insights.main.name
}

output "container_registry_name" {
  description = "Name of the Azure Container Registry used by Azure Machine Learning."
  value       = azurerm_container_registry.main.name
}

output "container_registry_login_server" {
  description = "Login server for the Azure Container Registry."
  value       = azurerm_container_registry.main.login_server
}