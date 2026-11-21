# Escalation records -> Firehose (regional) -> S3, partitioned by region.
# With no central bucket given, records stay in this region.
resource "aws_s3_bucket" "records" {
  count         = var.records_bucket_arn == "" ? 1 : 0
  bucket_prefix = "${var.name}-records-${local.region}-"
  tags          = local.tags
}

resource "aws_s3_bucket_public_access_block" "records" {
  count                   = length(aws_s3_bucket.records)
  bucket                  = aws_s3_bucket.records[0].id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "records" {
  count  = length(aws_s3_bucket.records)
  bucket = aws_s3_bucket.records[0].id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "aws:kms" }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "records" {
  count  = length(aws_s3_bucket.records)
  bucket = aws_s3_bucket.records[0].id
  rule {
    id     = "tier-and-expire"
    status = "Enabled"
    filter { prefix = "" }
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
    expiration { days = 365 }
  }
}

locals {
  records_bucket_arn = var.records_bucket_arn != "" ? var.records_bucket_arn : aws_s3_bucket.records[0].arn
}

resource "aws_iam_role" "firehose" {
  name_prefix = "hei-fh-"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action    = "sts:AssumeRole"
      Principal = { Service = "firehose.amazonaws.com" }
    }]
  })
  tags = local.tags
}

resource "aws_iam_role_policy" "firehose" {
  role = aws_iam_role.firehose.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:AbortMultipartUpload", "s3:GetBucketLocation", "s3:GetObject", "s3:ListBucket", "s3:ListBucketMultipartUploads", "s3:PutObject"]
      Resource = [local.records_bucket_arn, "${local.records_bucket_arn}/*"]
    }]
  })
}

resource "aws_kinesis_firehose_delivery_stream" "records" {
  name        = "${local.prefix}-escalations"
  destination = "extended_s3"

  server_side_encryption { enabled = true }

  extended_s3_configuration {
    role_arn            = aws_iam_role.firehose.arn
    bucket_arn          = local.records_bucket_arn
    prefix              = "escalations/region=${local.region}/!{timestamp:yyyy/MM/dd/HH}/"
    error_output_prefix = "errors/region=${local.region}/!{firehose:error-output-type}/!{timestamp:yyyy/MM/dd}/"
    buffering_size      = 64
    buffering_interval  = 300
    compression_format  = "GZIP"
  }

  tags = local.tags
}

# The gateway emits CloudWatch EMF; alarm when this edge escalates too much
# (model regression, distribution shift, or a mis-set threshold -> cost spike).
resource "aws_cloudwatch_metric_alarm" "escalation_rate" {
  alarm_name          = "${local.prefix}-escalation-rate"
  alarm_description   = "Escalation rate to Claude above ${var.escalation_rate_alarm} for 15 minutes"
  comparison_operator = "GreaterThanThreshold"
  threshold           = var.escalation_rate_alarm
  evaluation_periods  = 3
  treat_missing_data  = "notBreaching"

  metric_query {
    id          = "rate"
    expression  = "IF(req > 20, esc / req, 0)"
    label       = "Escalation rate"
    return_data = true
  }

  metric_query {
    id = "esc"
    metric {
      namespace   = "HybridEdge"
      metric_name = "Escalations"
      dimensions  = { Region = local.region }
      period      = 300
      stat        = "Sum"
    }
  }

  metric_query {
    id = "req"
    metric {
      namespace   = "HybridEdge"
      metric_name = "Requests"
      dimensions  = { Region = local.region }
      period      = 300
      stat        = "Sum"
    }
  }

  tags = local.tags
}
