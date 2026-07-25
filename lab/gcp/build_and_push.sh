#!/usr/bin/env bash
# Build and push the lab/gcp runner image to Artifact Registry.
#
# You rebuild this image ONLY when dependencies (or the base image) change.
# Code changes ship via `git push` + resubmit; see lab/gcp/README.md.
#
# Config via env or flags:
#   PROJECT_ID     GCP project. Default: gcloud config core/project.
#   REGION         Artifact Registry region. Default: gcloud config compute/region.
#   AR_REPOSITORY  Docker repo name. Default: verl-rl.
#   IMAGE_NAME     Image name. Default: verl-gcp-runner.
#   IMAGE_TAG      Tag. Default: runner.
#   BASE_IMAGE     Base image. Default: verlai/verl:vllm012.latest.
#   DOCKER_PLATFORM Target platform. Default: linux/amd64 (required for GCP GPUs).
#
# Usage:
#   lab/gcp/build_and_push.sh [--check] [--tag TAG] [--no-create-repo]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() { sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

CHECK_ONLY=0
CREATE_REPO=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) CHECK_ONLY=1; shift ;;
    --no-create-repo) CREATE_REPO=0; shift ;;
    --tag) IMAGE_TAG="${2:?missing tag after --tag}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value core/project 2>/dev/null || true)}"
REGION="${REGION:-$(gcloud config get-value compute/region 2>/dev/null || true)}"
AR_REPOSITORY="${AR_REPOSITORY:-verl-rl}"
IMAGE_NAME="${IMAGE_NAME:-verl-gcp-runner}"
IMAGE_TAG="${IMAGE_TAG:-runner}"
BASE_IMAGE="${BASE_IMAGE:-verlai/verl:vllm012.latest}"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"

PROJECT_ID="${PROJECT_ID//[$'\r\n']}"
REGION="${REGION//[$'\r\n']}"

IMAGE_URI="${IMAGE_URI:-${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}}"

cat <<EOF
Resolved build settings
  PROJECT_ID:    ${PROJECT_ID:-<missing>}
  REGION:        ${REGION:-<missing>}
  AR_REPOSITORY: ${AR_REPOSITORY}
  IMAGE_NAME:    ${IMAGE_NAME}
  IMAGE_TAG:     ${IMAGE_TAG}
  IMAGE_URI:     ${IMAGE_URI}
  BASE_IMAGE:    ${BASE_IMAGE}
  PLATFORM:      ${DOCKER_PLATFORM}
  BUILD_CONTEXT: ${REPO_ROOT}
EOF

missing=()
[[ -n "${PROJECT_ID}" ]] || missing+=(PROJECT_ID)
[[ -n "${REGION}" ]] || missing+=(REGION)
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "Missing required values: ${missing[*]}" >&2
  echo "Set via env/flags or gcloud config." >&2
  exit 1
fi

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
  echo
  command -v gcloud >/dev/null && echo "gcloud: $(command -v gcloud)" || echo "gcloud: MISSING"
  command -v docker >/dev/null && echo "docker: $(command -v docker)" || echo "docker: MISSING"
  docker info >/dev/null 2>&1 && echo "docker daemon: reachable" || echo "docker daemon: NOT reachable (start Docker Desktop)"
  exit 0
fi

if ! gcloud artifacts repositories describe "${AR_REPOSITORY}" \
    --project="${PROJECT_ID}" --location="${REGION}" >/dev/null 2>&1; then
  if [[ "${CREATE_REPO}" -eq 1 ]]; then
    echo "Creating Artifact Registry repo ${AR_REPOSITORY} in ${REGION}"
    gcloud artifacts repositories create "${AR_REPOSITORY}" \
      --project="${PROJECT_ID}" --location="${REGION}" \
      --repository-format=docker --description="verl RL training images"
  else
    echo "Artifact Registry repo not found: ${AR_REPOSITORY} (${REGION})" >&2
    exit 1
  fi
fi

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# Build context is the repo root so the Dockerfile can COPY lab/gcp/bootstrap.sh.
docker build \
  --platform "${DOCKER_PLATFORM}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "${IMAGE_URI}" \
  "${REPO_ROOT}"

docker push "${IMAGE_URI}"
echo "Pushed ${IMAGE_URI}"
