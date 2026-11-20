variable "name" {
  type    = string
  default = "hybrid-edge"
}

variable "home_region" {
  type    = string
  default = "us-east-1"
}

variable "edge_regions" {
  description = "Edge regions to deploy. Each must have a provider alias + module in edges.tf."
  type        = list(string)
  default     = ["us-east-1", "eu-west-1", "ap-south-1"]
  validation {
    condition     = length(setsubtract(var.edge_regions, ["us-east-1", "eu-west-1", "ap-south-1", "ap-southeast-1"])) == 0
    error_message = "Supported edge regions: us-east-1, eu-west-1, ap-south-1, ap-southeast-1 (add more in versions.tf + edges.tf)."
  }
}

variable "gateway_image_tag" {
  type    = string
  default = "latest"
}

variable "vllm_image" {
  type    = string
  default = "vllm/vllm-openai:v0.6.6"
}

variable "edge_model" {
  type    = string
  default = "Qwen/Qwen2.5-7B-Instruct"
}

variable "instance_type" {
  type    = string
  default = "g6.xlarge"
}

variable "min_nodes" {
  type    = number
  default = 1
}

variable "max_nodes" {
  type    = number
  default = 4
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
  description = "anthropic (Claude API, key in Secrets Manager) or bedrock (Claude on Amazon Bedrock, IAM auth)."
  type        = string
  default     = "anthropic"
  validation {
    condition     = contains(["anthropic", "bedrock"], var.cloud_provider)
    error_message = "cloud_provider must be anthropic or bedrock."
  }
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
  type    = number
  default = 600
}

variable "enable_hf_token" {
  description = "Create a Secrets Manager secret for a Hugging Face token (needed for gated models such as Llama)."
  type        = bool
  default     = false
}

variable "enable_gateway_api_keys" {
  description = "Create a secret holding comma-separated client API keys; the gateway then requires one."
  type        = bool
  default     = true
}

variable "record_include_text" {
  type    = bool
  default = true
}

variable "certificate_arns" {
  description = "Optional map of region -> ACM certificate ARN for HTTPS on each ALB."
  type        = map(string)
  default     = {}
}

variable "ingress_cidrs" {
  type    = list(string)
  default = ["0.0.0.0/0"]
}

variable "manage_ecr_replication" {
  description = "Manage the account's (singleton) ECR replication configuration. Disable if you manage it elsewhere."
  type        = bool
  default     = true
}

variable "tags" {
  type    = map(string)
  default = { project = "hybrid-edge-intelligence" }
}
