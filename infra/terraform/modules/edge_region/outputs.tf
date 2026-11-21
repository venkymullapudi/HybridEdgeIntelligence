output "region" {
  value = local.region
}

output "alb_arn" {
  value = aws_lb.this.arn
}

output "alb_dns_name" {
  value = aws_lb.this.dns_name
}

output "cluster_name" {
  value = aws_ecs_cluster.this.name
}

output "firehose_stream" {
  value = aws_kinesis_firehose_delivery_stream.records.name
}

output "log_group" {
  value = aws_cloudwatch_log_group.this.name
}

output "records_bucket_arn" {
  value = local.records_bucket_arn
}
