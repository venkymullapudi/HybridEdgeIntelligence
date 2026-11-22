#!/usr/bin/env bash
# Build the gateway image, push it to ECR (which replicates it to every edge
# region), roll the ECS services in all edges, and verify each one actually runs
# the new image.
#
#   scripts/deploy.sh [--yes] [image-tag]
#
#   --yes   apply without the interactive Terraform review (for CI). Without it,
#           Terraform shows the full plan and asks: a deploy applies *every*
#           pending infrastructure change, not just the image tag.
#
# Env: TF_BIN (default terraform), REPLICATION_TIMEOUT_S (900), POLL_S (10).
# Requires: aws CLI, docker (buildx), python3, terraform/tofu already applied once.
set -euo pipefail

AUTO_APPROVE=""
TAG=""
for arg in "$@"; do
  case "$arg" in
    --yes) AUTO_APPROVE=1 ;;
    -*) echo "unknown option: $arg" >&2; exit 2 ;;
    *) TAG="$arg" ;;
  esac
done
TAG="${TAG:-$(git rev-parse --short HEAD)}"
TF_DIR="$(cd "$(dirname "$0")/../infra/terraform" && pwd)"
TF="${TF_BIN:-terraform}"
REPLICATION_TIMEOUT_S="${REPLICATION_TIMEOUT_S:-900}"
POLL_S="${POLL_S:-10}"

REPO_URL="$($TF -chdir="$TF_DIR" output -raw ecr_repository_url)"
HOME_REGION="$(echo "$REPO_URL" | cut -d. -f4)"
REGISTRY="${REPO_URL%%/*}"
REPO_NAME="${REPO_URL#*/}"
EDGES_JSON="$($TF -chdir="$TF_DIR" output -json edges)"
REGIONS="$(python3 -c 'import json,sys; print(" ".join(sorted(json.load(sys.stdin))))' <<<"$EDGES_JSON")"
cluster_of() {
  python3 -c 'import json,sys; print(json.load(sys.stdin)[sys.argv[1]]["cluster"])' "$1" <<<"$EDGES_JSON"
}

aws ecr get-login-password --region "$HOME_REGION" | docker login --username AWS --password-stdin "$REGISTRY"
docker buildx build --platform linux/amd64 -f docker/Dockerfile -t "$REPO_URL:$TAG" -t "$REPO_URL:latest" --push .
echo "Pushed $REPO_URL:$TAG"

# Wait until the tag exists in every edge region: a region the image hasn't reached
# yet fails to pull, and ECS rolls it back while the rest of the deploy "succeeds".
deadline=$((SECONDS + REPLICATION_TIMEOUT_S))
for r in $REGIONS; do
  until aws ecr describe-images --region "$r" --repository-name "$REPO_NAME" \
          --image-ids imageTag="$TAG" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "image $TAG did not replicate to $r within ${REPLICATION_TIMEOUT_S}s" >&2
      exit 1
    fi
    echo "waiting for $TAG to replicate to $r..."
    sleep "$POLL_S"
  done
  echo "replicated to $r"
done

if [[ -n "$AUTO_APPROVE" ]]; then
  $TF -chdir="$TF_DIR" apply -auto-approve -var "gateway_image_tag=$TAG"
else
  $TF -chdir="$TF_DIR" apply -var "gateway_image_tag=$TAG"
fi

# Verify each edge: wait for the rollout, then check the service runs the new image.
# A deployment the circuit breaker rolled back is "stable" too, on the old image.
failed=""
for r in $REGIONS; do
  cluster="$(cluster_of "$r")"
  echo "waiting for $r ($cluster) to stabilise..."
  if ! aws ecs wait services-stable --region "$r" --cluster "$cluster" --services edge; then
    failed="$failed $r(not-stable)"
    continue
  fi
  td="$(aws ecs describe-services --region "$r" --cluster "$cluster" --services edge \
          --query 'services[0].taskDefinition' --output text)"
  image="$(aws ecs describe-task-definition --region "$r" --task-definition "$td" \
          --query "taskDefinition.containerDefinitions[?name=='gateway'].image" --output text)"
  if [[ "$image" != *":$TAG" ]]; then
    failed="$failed $r(running:${image##*:})"
  else
    echo "$r runs $TAG"
  fi
done

if [[ -n "$failed" ]]; then
  echo "deploy of $TAG FAILED in:$failed (likely rolled back by the ECS deployment circuit breaker; check the service events)" >&2
  exit 1
fi
echo "deploy of $TAG complete in: $REGIONS"
