provider "azurerm" {
  features {
    # Application Insights creates supporting resources outside Terraform state.
    # Allow this dedicated development resource group to be removed cleanly
    # during infrastructure teardown.
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
}