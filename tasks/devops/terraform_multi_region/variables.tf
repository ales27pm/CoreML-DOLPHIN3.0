variable "primary_region" {
  description = "AWS region hosting the primary stack"
  type        = string
}

variable "secondary_region" {
  description = "AWS region hosting the failover stack"
  type        = string
}

variable "service_name" {
  description = "Base name used for tagging and resource identification"
  type        = string
  default     = "dolphin-platform"
}

variable "environment" {
  description = "Deployment environment identifier (e.g., production, staging)"
  type        = string
  default     = "production"
}

variable "health_check_fqdn" {
  description = "Fully-qualified domain name monitored by Route53 health checks"
  type        = string
  default     = "api.example.com"
}
