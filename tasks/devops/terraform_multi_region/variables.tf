variable "primary_region" {
  description = "AWS region hosting the primary stack"
  type        = string
  validation {
    condition     = can(regex("^[a-z]{2}-[a-z0-9-]+-\\d$", var.primary_region))
    error_message = "primary_region must be a valid AWS region identifier (e.g., us-east-1)."
  }
}

variable "secondary_region" {
  description = "AWS region hosting the failover stack"
  type        = string
  validation {
    condition     = can(regex("^[a-z]{2}-[a-z0-9-]+-\\d$", var.secondary_region))
    error_message = "secondary_region must be a valid AWS region identifier (e.g., us-west-2)."
  }
}

variable "service_name" {
  description = "Base name used for tagging and resource identification"
  type        = string
  default     = "dolphin-platform"
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,32}$", var.service_name))
    error_message = "service_name must start with a letter and contain only lowercase letters, numbers, or hyphens."
  }
}

variable "environment" {
  description = "Deployment environment identifier (e.g., production, staging)"
  type        = string
  default     = "production"
  validation {
    condition     = contains(["dev", "staging", "prod", "production"], var.environment)
    error_message = "environment must be one of dev, staging, prod, or production."
  }
}

variable "health_check_fqdn" {
  description = "Fully-qualified domain name monitored by Route53 health checks"
  type        = string
  default     = "api.example.com"
  validation {
    condition     = can(regex("^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*$", var.health_check_fqdn))
    error_message = "health_check_fqdn must be a valid hostname."
  }
}

variable "tls_certificate_arn" {
  description = "ACM certificate ARN used by the application load balancers"
  type        = string
  nullable    = false
  validation {
    condition     = can(regex("^arn:aws:acm:[a-z0-9-]+:\\d{12}:certificate/.+", var.tls_certificate_arn))
    error_message = "tls_certificate_arn must reference a valid ACM certificate ARN."
  }
}
