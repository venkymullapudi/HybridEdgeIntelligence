#!/usr/bin/env bash
# Build the gateway image, push it to ECR (which replicates it to every edge
# region), and roll the ECS services in all edges.
#
#   scripts/deploy.sh [image-tag]
#
# Requires: aws CLI, docker (buildx), terraform/tofu already applied once.
set -euo pipefail

TAG="${1:-$(git rev-parse --short HEAD)}"
TF_DIR="$(cd "$(dirname "$0")/../infra/terraform" && pwd)"
TF="${TF_BIN:-terraform}"

REPO_URL="$($TF -chdir="$TF_DIR" output -raw ecr_repository_url)"
HOME_REGION="$(echo "$REPO_URL" | cut -d. -f4)"
REGISTRY="${REPO_URL%%/*}"

aws ecr get-login-password --region "$HOME_REGION" | docker login --username AWS --password-stdin "$REGISTRY"
docker buildx build --platform linux/amd64 -f docker/Dockerfile -t "$REPO_URL:$TAG" -t "$REPO_URL:latest" --push .

echo "Pushed $REPO_URL:$TAG; waiting for cross-region replication..."
sleep 60

# Point every edge at the new tag and let ECS do a rolling, health-checked deploy.
$TF -chdir="$TF_DIR" apply -auto-approve -var "gateway_image_tag=$TAG"
