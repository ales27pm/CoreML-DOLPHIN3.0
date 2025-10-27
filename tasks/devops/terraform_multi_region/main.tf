terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

provider "aws" {
  region = var.primary_region
}

provider "aws" {
  alias  = "secondary"
  region = var.secondary_region
}

module "primary" {
  source    = "./modules/app"
  providers = {
    aws = aws
  }

  region             = var.primary_region
  service_name       = var.service_name
  environment        = var.environment
  enable_failover    = false
  health_check_fqdn  = var.health_check_fqdn
  tls_certificate_arn = var.tls_certificate_arn
}

module "secondary" {
  source    = "./modules/app"
  providers = {
    aws = aws.secondary
  }

  region             = var.secondary_region
  service_name       = "${var.service_name}-secondary"
  environment        = var.environment
  enable_failover    = true
  health_check_fqdn  = var.health_check_fqdn
  tls_certificate_arn = var.tls_certificate_arn

  depends_on = [module.primary]
}

output "primary_endpoint" {
  value       = module.primary.alb_dns_name
  description = "DNS name for the primary region application load balancer."
}

output "secondary_endpoint" {
  value       = module.secondary.alb_dns_name
  description = "DNS name for the secondary region failover load balancer."
}

output "route53_health_checks" {
  value       = concat(module.primary.health_check_ids, module.secondary.health_check_ids)
  description = "Identifiers of Route53 health checks monitoring the ALBs."
}
