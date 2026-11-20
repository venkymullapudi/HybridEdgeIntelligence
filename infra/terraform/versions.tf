terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 6.0, < 7.0"
    }
  }
}

# Home region: ECR, the records bucket and the primary copy of secrets.
provider "aws" {
  region = var.home_region
  default_tags { tags = var.tags }
}

# One aliased provider per edge region. To add an edge (e.g. sa-east-1), add a
# provider alias here and a module block in edges.tf.
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"
  default_tags { tags = var.tags }
}

provider "aws" {
  alias  = "eu_west_1"
  region = "eu-west-1"
  default_tags { tags = var.tags }
}

provider "aws" {
  alias  = "ap_south_1"
  region = "ap-south-1"
  default_tags { tags = var.tags }
}

provider "aws" {
  alias  = "ap_southeast_1"
  region = "ap-southeast-1"
  default_tags { tags = var.tags }
}

# Global Accelerator's control plane lives in us-west-2.
provider "aws" {
  alias  = "global"
  region = "us-west-2"
  default_tags { tags = var.tags }
}
