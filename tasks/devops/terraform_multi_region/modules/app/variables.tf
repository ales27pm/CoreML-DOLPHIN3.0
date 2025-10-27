variable "region" {
  description = "AWS region used for resources"
  type        = string
  validation {
    condition     = can(regex("^[a-z]{2}-[a-z0-9-]+-\\d$", var.region))
    error_message = "region must be a valid AWS region identifier (for example, us-east-1)."
  }
}

variable "service_name" {
  description = "Service identifier for tagging"
  type        = string
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,32}$", var.service_name))
    error_message = "service_name must start with a letter and contain only lowercase letters, digits, or hyphens."
  }
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod", "production"], var.environment)
    error_message = "environment must be one of dev, staging, prod, or production."
  }
}

variable "enable_failover" {
  description = "Whether Route53 health checks should be provisioned for failover"
  type        = bool
}

variable "health_check_fqdn" {
  description = "FQDN monitored by Route53"
  type        = string
  validation {
    condition     = can(regex("^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*$", var.health_check_fqdn))
    error_message = "health_check_fqdn must be a valid hostname."
  }
}

variable "tls_certificate_arn" {
  description = "ARN of the ACM certificate attached to the load balancer"
  type        = string
  nullable    = false
  validation {
    condition     = can(regex("^arn:aws:acm:[a-z0-9-]+:\\d{12}:certificate/.+", var.tls_certificate_arn))
    error_message = "tls_certificate_arn must reference a valid ACM certificate ARN."
  }
}
