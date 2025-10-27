variable "region" {
  description = "AWS region used for resources"
  type        = string
}

variable "service_name" {
  description = "Service identifier for tagging"
  type        = string
}

variable "environment" {
  description = "Deployment environment"
  type        = string
}

variable "enable_failover" {
  description = "Whether Route53 health checks should be provisioned for failover"
  type        = bool
}

variable "health_check_fqdn" {
  description = "FQDN monitored by Route53"
  type        = string
}

variable "tls_certificate_arn" {
  description = "ARN of the ACM certificate attached to the load balancer"
  type        = string
  default     = "arn:aws:acm:us-east-1:123456789012:certificate/example"
}
