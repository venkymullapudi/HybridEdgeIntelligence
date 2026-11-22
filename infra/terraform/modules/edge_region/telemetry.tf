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

# Records hold prompts and answers: refuse any access without TLS.
resource "aws_s3_bucket_policy" "records_tls_only" {
  count  = length(aws_s3_bucket.records)
  bucket = aws_s3_bucket.records[0].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "DenyInsecureTransport"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource  = [aws_s3_bucket.records[0].arn, "${aws_s3_bucket.records[0].arn}/*"]
      Condition = { Bool = { "aws:SecureTransport" = "false" } }
    }]
  })
  depends_on = [aws_s3_bucket_public_access_block.records]
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
# Alarms notify this regional topic (alarm actions must be in the alarm's region).
# Not encrypted with the AWS-managed aws/sns key: its key policy doesn't let
# CloudWatch publish, so alarm notifications would be silently dropped. Use a
# customer-managed key that grants cloudwatch.amazonaws.com if you need SSE.
resource "aws_sns_topic" "alarms" {
  name = "${local.prefix}-alarms"
  tags = local.tags
}

resource "aws_sns_topic_subscription" "alarm_email" {
  for_each  = toset(var.alarm_emails)
  topic_arn = aws_sns_topic.alarms.arn
  protocol  = "email" # each address must confirm the subscription email
  endpoint  = each.value
}

resource "aws_cloudwatch_metric_alarm" "escalation_rate" {
  alarm_name          = "${local.prefix}-escalation-rate"
  alarm_description   = "Escalation rate to Claude above ${var.escalation_rate_alarm} for 15 minutes"
  comparison_operator = "GreaterThanThreshold"
  threshold           = var.escalation_rate_alarm
  evaluation_periods  = 3
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
  ok_actions          = [aws_sns_topic.alarms.arn]

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

# Degraded answers are what users feel when Claude is down, throttled or over budget.
resource "aws_cloudwatch_metric_alarm" "degraded_rate" {
  alarm_name          = "${local.prefix}-degraded-rate"
  alarm_description   = "More than ${var.degraded_rate_alarm * 100}% of answers degraded (cloud unavailable) for 15 minutes; see hybrid_edge_cloud_unavailable_total{reason} for why"
  comparison_operator = "GreaterThanThreshold"
  threshold           = var.degraded_rate_alarm
  evaluation_periods  = 3
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
  ok_actions          = [aws_sns_topic.alarms.arn]

  metric_query {
    id          = "rate"
    expression  = "IF(req > 20, deg / req, 0)"
    label       = "Degraded rate"
    return_data = true
  }

  metric_query {
    id = "deg"
    metric {
      namespace   = "HybridEdge"
      metric_name = "Degraded"
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
