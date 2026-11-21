variable "name" {
  description = "Deployment name prefix."
  type        = string
}

variable "vpc_cidr" {
  type    = string
  default = "10.40.0.0/16"
}

variable "gateway_image" {
  description = "Gateway container image URI (ECR, replicated into this region)."
  type        = string
}

variable "vllm_image" {
  type    = string
  default = "vllm/vllm-openai:v0.6.6"
}

variable "edge_model" {
  description = "Hugging Face id of the open-weight model served at the edge."
  type        = string
}

variable "edge_max_model_len" {
  type    = number
  default = 8192
}

variable "instance_type" {
  description = "GPU instance type. g6.xlarge (1x L4 24GB) fits 7-8B models in bf16."
  type        = string
  default     = "g6.xlarge"
}

variable "min_nodes" {
  type    = number
  default = 1
}

variable "max_nodes" {
  type    = number
  default = 4
}

variable "desired_tasks" {
  type    = number
  default = 1
}

variable "gate_threshold" {
  type    = number
  default = 0.7
}

variable "gate_consistency_samples" {
  type    = number
  default = 0
}

variable "cloud_provider" {
  description = "anthropic | bedrock"
  type        = string
  default     = "anthropic"
}

variable "cloud_model" {
  type    = string
  default = "claude-opus-5-5"
}

variable "cloud_effort" {
  type    = string
  default = "medium"
}

variable "max_escalations_per_min" {
  description = "Per-task cost guard for escalations."
  type        = number
  default     = 600
}

variable "anthropic_secret_arn" {
  description = "Secrets Manager ARN (in this region) holding the Anthropic API key. Empty for Bedrock."
  type        = string
  default     = ""
}

variable "hf_token_secret_arn" {
  description = "Optional Secrets Manager ARN (in this region) with a Hugging Face token for gated models."
  type        = string
  default     = ""
}

variable "gateway_api_keys_secret_arn" {
  description = "Optional Secrets Manager ARN with comma-separated client API keys for the gateway."
  type        = string
  default     = ""
}

variable "records_bucket_arn" {
  description = "S3 bucket for escalation records. Empty = create a bucket in this region (data residency)."
  type        = string
  default     = ""
}

variable "record_include_text" {
  type    = bool
  default = true
}

variable "certificate_arn" {
  description = "ACM certificate for HTTPS on the ALB. Empty = HTTP only (put TLS in front another way)."
  type        = string
  default     = ""
}

variable "ingress_cidrs" {
  description = "CIDRs allowed to reach the ALB. Global Accelerator preserves client IPs."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "escalation_rate_alarm" {
  description = "Alarm when the escalation rate stays above this fraction."
  type        = number
  default     = 0.5
}

variable "tags" {
  type    = map(string)
  default = {}
}

variable "shadow_rate" {
  type    = number
  default = 0.0
}

variable "max_escalations_per_key_per_min" {
  type    = number
  default = 120
}

variable "cloud_aws_region" {
  description = "Bedrock region to call. Empty = this edge's own region."
  type        = string
  default     = ""
}
