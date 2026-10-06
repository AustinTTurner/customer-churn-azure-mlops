variable "storage_account_replication_type" {
  description = "Replication strategy for the Azure Storage Account."
  type        = string
  default     = "LRS"

  validation {
    condition = contains(
      ["LRS", "GRS", "RAGRS", "ZRS", "GZRS", "RAGZRS"],
      var.storage_account_replication_type
    )

    error_message = "Storage replication type is not supported."
  }
}

variable "container_registry_sku" {
  description = "SKU for the Azure Container Registry used by Azure Machine Learning."
  type        = string
  default     = "Basic"

  validation {
    condition = contains(
      ["Basic", "Standard", "Premium"],
      var.container_registry_sku
    )

    error_message = "Container Registry SKU must be Basic, Standard, or Premium."
  }
}

variable "log_analytics_retention_days" {
  description = "Number of days to retain Log Analytics workspace data."
  type        = number
  default     = 30

  validation {
    condition = (
      var.log_analytics_retention_days >= 30 &&
      var.log_analytics_retention_days <= 730
    )

    error_message = "Log Analytics retention must be between 30 and 730 days."
  }
}

variable "key_vault_soft_delete_retention_days" {
  description = "Number of days to retain soft-deleted Key Vault resources."
  type        = number
  default     = 7

  validation {
    condition = (
      var.key_vault_soft_delete_retention_days >= 7 &&
      var.key_vault_soft_delete_retention_days <= 90
    )

    error_message = "Key Vault soft-delete retention must be between 7 and 90 days."
  }
}