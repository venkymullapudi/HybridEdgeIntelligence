data "aws_caller_identity" "current" {}

locals {
  account      = data.aws_caller_identity.current.account_id
  use_tls      = length(var.certificate_arns) > 0
  other_region = [for r in var.edge_regions : r if r != var.home_region]
}

# --------------------------------------------------------------------------
# Container registry (home region) replicated into every edge region
# --------------------------------------------------------------------------
resource "aws_ecr_repository" "gateway" {
  name                 = "${var.name}/gateway"
  image_tag_mutability = "MUTABLE"
  force_delete         = false
  image_scanning_configuration { scan_on_push = true }
  encryption_configuration { encryption_type = "AES256" }
}

resource "aws_ecr_replication_configuration" "this" {
  count = var.manage_ecr_replication && length(local.other_region) > 0 ? 1 : 0
  replication_configuration {
    rule {
      dynamic "destination" {
        for_each = local.other_region
        content {
          region      = destination.value
          registry_id = local.account
        }
      }
      repository_filter {
        filter      = "${var.name}/"
        filter_type = "PREFIX_MATCH"
      }
    }
  }
}

# --------------------------------------------------------------------------
# Secrets: created empty, replicated to every edge region. Set values out of
# band so keys never land in Terraform state:
#   aws secretsmanager put-secret-value --secret-id <name> --secret-string <value>
# --------------------------------------------------------------------------
resource "aws_secretsmanager_secret" "anthropic" {
  count                   = var.cloud_provider == "anthropic" ? 1 : 0
  name                    = "${var.name}/anthropic-api-key"
  description             = "Anthropic API key used by edge gateways for escalations"
  recovery_window_in_days = 7
  dynamic "replica" {
    for_each = local.other_region
    content { region = replica.value }
  }
}

resource "aws_secretsmanager_secret" "hf_token" {
  count                   = var.enable_hf_token ? 1 : 0
  name                    = "${var.name}/hf-token"
  description             = "Hugging Face token for gated open-weight models"
  recovery_window_in_days = 7
  dynamic "replica" {
    for_each = local.other_region
    content { region = replica.value }
  }
}

resource "aws_secretsmanager_secret" "gateway_api_keys" {
  count                   = var.enable_gateway_api_keys ? 1 : 0
  name                    = "${var.name}/gateway-api-keys"
  description             = "Comma-separated client API keys accepted by edge gateways"
  recovery_window_in_days = 7
  dynamic "replica" {
    for_each = local.other_region
    content { region = replica.value }
  }
}

# --------------------------------------------------------------------------
# Central bucket for escalation records: the data flywheel for distilling
# Claude's answers back into the edge model and recalibrating the gate.
# --------------------------------------------------------------------------
resource "aws_s3_bucket" "records" {
  bucket_prefix = "${var.name}-records-"
}

resource "aws_s3_bucket_public_access_block" "records" {
  bucket                  = aws_s3_bucket.records.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "records" {
  bucket = aws_s3_bucket.records.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "aws:kms" }
  }
}

resource "aws_s3_bucket_versioning" "records" {
  bucket = aws_s3_bucket.records.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_lifecycle_configuration" "records" {
  bucket = aws_s3_bucket.records.id
  rule {
    id     = "tier-and-expire"
    status = "Enabled"
    filter { prefix = "" }
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
    expiration { days = 365 }
    noncurrent_version_expiration { noncurrent_days = 30 }
  }
}

locals {
  # A replica keeps the primary's ARN suffix; only the region differs.
  secret_arn_in = {
    for r in var.edge_regions : r => {
      anthropic = length(aws_secretsmanager_secret.anthropic) > 0 ? replace(aws_secretsmanager_secret.anthropic[0].arn, ":${var.home_region}:", ":${r}:") : ""
      hf        = length(aws_secretsmanager_secret.hf_token) > 0 ? replace(aws_secretsmanager_secret.hf_token[0].arn, ":${var.home_region}:", ":${r}:") : ""
      api_keys  = length(aws_secretsmanager_secret.gateway_api_keys) > 0 ? replace(aws_secretsmanager_secret.gateway_api_keys[0].arn, ":${var.home_region}:", ":${r}:") : ""
    }
  }

  gateway_image_in = {
    for r in var.edge_regions : r => "${local.account}.dkr.ecr.${r}.amazonaws.com/${aws_ecr_repository.gateway.name}:${var.gateway_image_tag}"
  }
}
