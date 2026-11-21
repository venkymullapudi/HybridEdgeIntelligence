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

variable "allow_insecure_http" {
  description = "Allow deploying without TLS (HTTP on port 80). Client API keys then travel in cleartext."
  type        = bool
  default     = false
}

variable "records_residency" {
  description = "regional: each edge keeps its escalation records in a bucket in its own region (data stays in-region). central: all edges write to one bucket in home_region."
  type        = string
  default     = "regional"
  validation {
    condition     = contains(["regional", "central"], var.records_residency)
    error_message = "records_residency must be regional or central."
  }
}

variable "shadow_rate" {
  description = "Fraction of edge-served requests also sent to Claude in the background, to collect calibration labels."
  type        = number
  default     = 0.01
}

variable "max_escalations_per_key_per_min" {
  type    = number
  default = 120
}

variable "bedrock_region_overrides" {
  description = "With cloud_provider = bedrock: edge region -> Bedrock region to call, for edges whose own region doesn't offer the Claude model (e.g. { \"ap-south-1\" = \"ap-southeast-1\" }). Unlisted edges call Bedrock in their own region."
  type        = map(string)
  default     = {}
}

variable "edge_system_role" {
  description = "native, or merge for edge models whose chat template rejects a system role (e.g. Gemma)."
  type        = string
  default     = "native"
  validation {
    condition     = contains(["native", "merge"], var.edge_system_role)
    error_message = "edge_system_role must be native or merge."
  }
}
