#!/usr/bin/env bash
set -Eeuo pipefail

ENV_FILE="${1:-./ems_mvp.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Env file not found: ${ENV_FILE}" >&2
  exit 1
fi
source "${ENV_FILE}"

if [[ -z "${AWS_ACCOUNT_ID:-}" ]]; then echo "AWS_ACCOUNT_ID required"; exit 1; fi
if [[ -z "${AWS_REGION:-}" ]]; then echo "AWS_REGION required"; exit 1; fi
if [[ -z "${ECR_REPOSITORY:-}" ]]; then echo "ECR_REPOSITORY required"; exit 1; fi
if [[ -z "${IMAGE_TAG:-}" ]]; then echo "IMAGE_TAG required"; exit 1; fi
if [[ -z "${APP_NAME:-}" ]]; then echo "APP_NAME required"; exit 1; fi

IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}"

echo "Login ECR..."
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "Building image..."
docker build -t "${APP_NAME}:${IMAGE_TAG}" "${DOCKER_BUILD_CONTEXT:-.}"
docker tag "${APP_NAME}:${IMAGE_TAG}" "${IMAGE_URI}"

echo "Pushing image..."
docker push "${IMAGE_URI}"

echo "Helm upgrade/install..."
helm upgrade --install "${APP_NAME}" ./helm_chart -n "${K8S_NAMESPACE}" --set image.repository="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}" --set image.tag="${IMAGE_TAG}" --wait --timeout 300s

echo "Done."