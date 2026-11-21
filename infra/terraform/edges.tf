# One module per edge region. Terraform needs static provider references, so
# each region is its own block; `count` turns it on/off from var.edge_regions.

locals {
  edge_args = {
    name                     = var.name
    vllm_image               = var.vllm_image
    edge_model               = var.edge_model
    instance_type            = var.instance_type
    min_nodes                = var.min_nodes
    max_nodes                = var.max_nodes
    gate_threshold           = var.gate_threshold
    gate_consistency_samples = var.gate_consistency_samples
    cloud_provider           = var.cloud_provider
    cloud_model              = var.cloud_model
    cloud_effort             = var.cloud_effort
    max_escalations_per_min  = var.max_escalations_per_min
    record_include_text      = var.record_include_text
    ingress_cidrs            = var.ingress_cidrs
    shadow_rate              = var.shadow_rate
    edge_system_role         = var.edge_system_role
    per_key                  = var.max_escalations_per_key_per_min
    records_bucket_arn       = var.records_residency == "central" ? aws_s3_bucket.records[0].arn : ""
  }
}

module "edge_us_east_1" {
  source    = "./modules/edge_region"
  count     = contains(var.edge_regions, "us-east-1") ? 1 : 0
  providers = { aws = aws.us_east_1 }

  name                            = local.edge_args.name
  vllm_image                      = local.edge_args.vllm_image
  edge_model                      = local.edge_args.edge_model
  edge_system_role                = local.edge_args.edge_system_role
  instance_type                   = local.edge_args.instance_type
  min_nodes                       = local.edge_args.min_nodes
  max_nodes                       = local.edge_args.max_nodes
  gate_threshold                  = local.edge_args.gate_threshold
  gate_consistency_samples        = local.edge_args.gate_consistency_samples
  cloud_provider                  = local.edge_args.cloud_provider
  cloud_model                     = local.edge_args.cloud_model
  cloud_effort                    = local.edge_args.cloud_effort
  max_escalations_per_min         = local.edge_args.max_escalations_per_min
  record_include_text             = local.edge_args.record_include_text
  ingress_cidrs                   = local.edge_args.ingress_cidrs
  records_bucket_arn              = local.edge_args.records_bucket_arn
  shadow_rate                     = local.edge_args.shadow_rate
  max_escalations_per_key_per_min = local.edge_args.per_key
  gateway_image                   = local.gateway_image_in["us-east-1"]
  anthropic_secret_arn            = local.secret_arn_in["us-east-1"].anthropic
  hf_token_secret_arn             = local.secret_arn_in["us-east-1"].hf
  gateway_api_keys_secret_arn     = local.secret_arn_in["us-east-1"].api_keys
  certificate_arn                 = lookup(var.certificate_arns, "us-east-1", "")
  cloud_aws_region                = lookup(var.bedrock_region_overrides, "us-east-1", "")
  alarm_emails                    = var.alarm_emails
  vpc_cidr                        = "10.40.0.0/16"
}

module "edge_eu_west_1" {
  source    = "./modules/edge_region"
  count     = contains(var.edge_regions, "eu-west-1") ? 1 : 0
  providers = { aws = aws.eu_west_1 }

  name                            = local.edge_args.name
  vllm_image                      = local.edge_args.vllm_image
  edge_model                      = local.edge_args.edge_model
  edge_system_role                = local.edge_args.edge_system_role
  instance_type                   = local.edge_args.instance_type
  min_nodes                       = local.edge_args.min_nodes
  max_nodes                       = local.edge_args.max_nodes
  gate_threshold                  = local.edge_args.gate_threshold
  gate_consistency_samples        = local.edge_args.gate_consistency_samples
  cloud_provider                  = local.edge_args.cloud_provider
  cloud_model                     = local.edge_args.cloud_model
  cloud_effort                    = local.edge_args.cloud_effort
  max_escalations_per_min         = local.edge_args.max_escalations_per_min
  record_include_text             = local.edge_args.record_include_text
  ingress_cidrs                   = local.edge_args.ingress_cidrs
  records_bucket_arn              = local.edge_args.records_bucket_arn
  shadow_rate                     = local.edge_args.shadow_rate
  max_escalations_per_key_per_min = local.edge_args.per_key
  gateway_image                   = local.gateway_image_in["eu-west-1"]
  anthropic_secret_arn            = local.secret_arn_in["eu-west-1"].anthropic
  hf_token_secret_arn             = local.secret_arn_in["eu-west-1"].hf
  gateway_api_keys_secret_arn     = local.secret_arn_in["eu-west-1"].api_keys
  certificate_arn                 = lookup(var.certificate_arns, "eu-west-1", "")
  cloud_aws_region                = lookup(var.bedrock_region_overrides, "eu-west-1", "")
  alarm_emails                    = var.alarm_emails
  vpc_cidr                        = "10.41.0.0/16"
}

module "edge_ap_south_1" {
  source    = "./modules/edge_region"
  count     = contains(var.edge_regions, "ap-south-1") ? 1 : 0
  providers = { aws = aws.ap_south_1 }

  name                            = local.edge_args.name
  vllm_image                      = local.edge_args.vllm_image
  edge_model                      = local.edge_args.edge_model
  edge_system_role                = local.edge_args.edge_system_role
  instance_type                   = local.edge_args.instance_type
  min_nodes                       = local.edge_args.min_nodes
  max_nodes                       = local.edge_args.max_nodes
  gate_threshold                  = local.edge_args.gate_threshold
  gate_consistency_samples        = local.edge_args.gate_consistency_samples
  cloud_provider                  = local.edge_args.cloud_provider
  cloud_model                     = local.edge_args.cloud_model
  cloud_effort                    = local.edge_args.cloud_effort
  max_escalations_per_min         = local.edge_args.max_escalations_per_min
  record_include_text             = local.edge_args.record_include_text
  ingress_cidrs                   = local.edge_args.ingress_cidrs
  records_bucket_arn              = local.edge_args.records_bucket_arn
  shadow_rate                     = local.edge_args.shadow_rate
  max_escalations_per_key_per_min = local.edge_args.per_key
  gateway_image                   = local.gateway_image_in["ap-south-1"]
  anthropic_secret_arn            = local.secret_arn_in["ap-south-1"].anthropic
  hf_token_secret_arn             = local.secret_arn_in["ap-south-1"].hf
  gateway_api_keys_secret_arn     = local.secret_arn_in["ap-south-1"].api_keys
  certificate_arn                 = lookup(var.certificate_arns, "ap-south-1", "")
  cloud_aws_region                = lookup(var.bedrock_region_overrides, "ap-south-1", "")
  alarm_emails                    = var.alarm_emails
  vpc_cidr                        = "10.42.0.0/16"
}

module "edge_ap_southeast_1" {
  source    = "./modules/edge_region"
  count     = contains(var.edge_regions, "ap-southeast-1") ? 1 : 0
  providers = { aws = aws.ap_southeast_1 }

  name                            = local.edge_args.name
  vllm_image                      = local.edge_args.vllm_image
  edge_model                      = local.edge_args.edge_model
  edge_system_role                = local.edge_args.edge_system_role
  instance_type                   = local.edge_args.instance_type
  min_nodes                       = local.edge_args.min_nodes
  max_nodes                       = local.edge_args.max_nodes
  gate_threshold                  = local.edge_args.gate_threshold
  gate_consistency_samples        = local.edge_args.gate_consistency_samples
  cloud_provider                  = local.edge_args.cloud_provider
  cloud_model                     = local.edge_args.cloud_model
  cloud_effort                    = local.edge_args.cloud_effort
  max_escalations_per_min         = local.edge_args.max_escalations_per_min
  record_include_text             = local.edge_args.record_include_text
  ingress_cidrs                   = local.edge_args.ingress_cidrs
  records_bucket_arn              = local.edge_args.records_bucket_arn
  shadow_rate                     = local.edge_args.shadow_rate
  max_escalations_per_key_per_min = local.edge_args.per_key
  gateway_image                   = local.gateway_image_in["ap-southeast-1"]
  anthropic_secret_arn            = local.secret_arn_in["ap-southeast-1"].anthropic
  hf_token_secret_arn             = local.secret_arn_in["ap-southeast-1"].hf
  gateway_api_keys_secret_arn     = local.secret_arn_in["ap-southeast-1"].api_keys
  certificate_arn                 = lookup(var.certificate_arns, "ap-southeast-1", "")
  cloud_aws_region                = lookup(var.bedrock_region_overrides, "ap-southeast-1", "")
  alarm_emails                    = var.alarm_emails
  vpc_cidr                        = "10.43.0.0/16"
}

locals {
  edges = merge(
    { for m in module.edge_us_east_1 : m.region => m },
    { for m in module.edge_eu_west_1 : m.region => m },
    { for m in module.edge_ap_south_1 : m.region => m },
    { for m in module.edge_ap_southeast_1 : m.region => m },
  )
}

# --------------------------------------------------------------------------
# Global entry point: anycast IPs route each client to the nearest healthy
# edge. If an edge's /readyz fails (model down), traffic fails over to the
# next-closest region automatically.
# --------------------------------------------------------------------------
resource "aws_globalaccelerator_accelerator" "this" {
  provider        = aws.global
  name            = var.name
  ip_address_type = "IPV4"
  enabled         = true
}

resource "aws_globalaccelerator_listener" "this" {
  provider = aws.global

  lifecycle {
    precondition {
      condition     = local.use_tls || var.allow_insecure_http
      error_message = "No certificate_arns given. Provide one ACM certificate per edge region, or set allow_insecure_http = true to accept cleartext HTTP (API keys would travel unencrypted)."
    }
    precondition {
      condition     = !local.use_tls || length(setsubtract(var.edge_regions, keys(var.certificate_arns))) == 0
      error_message = "certificate_arns must cover every region in edge_regions when TLS is enabled."
    }
  }

  accelerator_arn = aws_globalaccelerator_accelerator.this.id
  protocol        = "TCP"
  client_affinity = "NONE"
  port_range {
    from_port = local.use_tls ? 443 : 80
    to_port   = local.use_tls ? 443 : 80
  }
}

resource "aws_globalaccelerator_endpoint_group" "edge" {
  provider                = aws.global
  for_each                = toset(var.edge_regions)
  listener_arn            = aws_globalaccelerator_listener.this.id
  endpoint_group_region   = each.key
  traffic_dial_percentage = 100
  health_check_protocol   = local.use_tls ? "HTTPS" : "HTTP"
  health_check_port       = local.use_tls ? 443 : 80
  health_check_path       = "/readyz"
  threshold_count         = 3

  endpoint_configuration {
    endpoint_id                    = local.edges[each.key].alb_arn
    weight                         = 100
    client_ip_preservation_enabled = true
  }
}
