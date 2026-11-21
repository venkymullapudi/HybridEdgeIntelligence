output "global_endpoint_dns" {
  description = "Single global hostname; clients are routed to the nearest healthy edge."
  value       = aws_globalaccelerator_accelerator.this.dns_name
}

output "global_endpoint_ips" {
  value = flatten([for s in aws_globalaccelerator_accelerator.this.ip_sets : s.ip_addresses])
}

output "edges" {
  value = {
    for r, m in local.edges : r => {
      alb_dns     = m.alb_dns_name
      cluster     = m.cluster_name
      firehose    = m.firehose_stream
      log_group   = m.log_group
      gateway_img = local.gateway_image_in[r]
    }
  }
}

output "ecr_repository_url" {
  value = aws_ecr_repository.gateway.repository_url
}

output "records_buckets" {
  description = "Where escalation records land, per edge region."
  value       = { for r, m in local.edges : r => m.records_bucket_arn }
}

output "secrets_to_populate" {
  value = compact([
    length(aws_secretsmanager_secret.anthropic) > 0 ? aws_secretsmanager_secret.anthropic[0].name : "",
    length(aws_secretsmanager_secret.hf_token) > 0 ? aws_secretsmanager_secret.hf_token[0].name : "",
    length(aws_secretsmanager_secret.gateway_api_keys) > 0 ? aws_secretsmanager_secret.gateway_api_keys[0].name : "",
  ])
}
