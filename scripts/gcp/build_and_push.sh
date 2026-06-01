#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=load_env.sh
source "${SCRIPT_DIR}/load_env.sh"

usage() {
  cat <<'EOF'
Usage: scripts/gcp/build_and_push.sh [--check] [--no-create-repo] [--tag TAG]

Builds the verl GCP training image and pushes it to Artifact Registry.

Options:
  --check           Print resolved values and missing setup, then exit.
  --no-create-repo Do not create the Artifact Registry repository if missing.
  --tag TAG         Override IMAGE_TAG for this build.

Configure with scripts/gcp/.env or exported env vars:
  PROJECT_ID       GCP project id. Defaults to `gcloud config get core/project`.
  REGION           Artifact Registry region, e.g. us-central1.
  AR_REPOSITORY    Artifact Registry Docker repo name. Defaults to verl-rl.
  IMAGE_NAME       Docker image name. Defaults to verl-gcp.
  IMAGE_TAG        Docker tag. Defaults to current git short SHA.
  IMAGE_URI        Optional full image URI; otherwise derived from the above.
  BASE_IMAGE       Optional base image. Defaults to verlai/verl:vllm012.latest.
  DOCKER_PLATFORM  Target image platform. Defaults to linux/amd64 for GCP GPUs.
EOF
}

CHECK_ONLY=0
CREATE_REPO=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)
      CHECK_ONLY=1
      shift
      ;;
    --no-create-repo)
      CREATE_REPO=0
      shift
      ;;
    --tag)
      IMAGE_TAG="${2:?missing tag after --tag}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

GCLOUD_PROJECT="$(gcloud config get-value core/project 2>/dev/null || true)"
GCLOUD_REGION="$(gcloud config get-value compute/region 2>/dev/null || true)"
GCLOUD_PROJECT="${GCLOUD_PROJECT//[$'\r\n']}"
GCLOUD_REGION="${GCLOUD_REGION//[$'\r\n']}"

PROJECT_ID="${PROJECT_ID:-${GCLOUD_PROJECT}}"
REGION="${REGION:-${GCLOUD_REGION}}"
AR_REPOSITORY="${AR_REPOSITORY:-verl-rl}"
IMAGE_NAME="${IMAGE_NAME:-verl-gcp}"

IMAGE_TAG="${IMAGE_TAG:-$(git -C "$(repo_root)" rev-parse --short HEAD)}"
if [[ -z "${IMAGE_URI:-}" && -n "${PROJECT_ID}" && -n "${REGION}" && -n "${AR_REPOSITORY}" && -n "${IMAGE_NAME}" && -n "${IMAGE_TAG}" ]]; then
  IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"
fi
BASE_IMAGE="${BASE_IMAGE:-verlai/verl:vllm012.latest}"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"

cd "$(repo_root)"

missing=()
for name in PROJECT_ID REGION AR_REPOSITORY IMAGE_NAME IMAGE_TAG BASE_IMAGE DOCKER_PLATFORM; do
  if [[ -z "${!name:-}" ]]; then
    missing+=("${name}")
  fi
done

cat <<EOF
Resolved build settings
  PROJECT_ID:    ${PROJECT_ID:-<missing>}
  REGION:        ${REGION:-<missing>}
  AR_REPOSITORY: ${AR_REPOSITORY:-<missing>}
  IMAGE_NAME:    ${IMAGE_NAME:-<missing>}
  IMAGE_TAG:     ${IMAGE_TAG:-<missing>}
  IMAGE_URI:     ${IMAGE_URI:-<missing>}
  BASE_IMAGE:    ${BASE_IMAGE:-<missing>}
  PLATFORM:      ${DOCKER_PLATFORM:-<missing>}
EOF

if [[ ${#missing[@]} -gt 0 ]]; then
  echo
  echo "Missing required values: ${missing[*]}"
  echo
  cat <<'EOF'
Create or choose these in GCP Console:
  1. A GCP project with billing enabled.
  2. A region for Artifact Registry, for example us-central1.
  3. A Docker Artifact Registry repository, for example verl-rl.
     Console path: Artifact Registry -> Repositories -> Create Repository
     Format: Docker
     Mode: Standard

Then set them in scripts/gcp/.env, for example:
  PROJECT_ID=your-project-id
  REGION=us-central1
  AR_REPOSITORY=verl-rl
  IMAGE_NAME=verl-gcp
EOF
  exit 1
fi

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
  echo
  echo "Local tool checks"
  command -v gcloud >/dev/null && echo "  gcloud: $(command -v gcloud)" || echo "  gcloud: missing"
  command -v docker >/dev/null && echo "  docker: $(command -v docker)" || echo "  docker: missing"
  docker info >/dev/null 2>&1 && echo "  docker daemon: reachable" || echo "  docker daemon: not reachable"
  echo
  cat <<EOF
GCP Console checklist
  Project exists and billing is enabled:
    ${PROJECT_ID}

  Artifact Registry API is enabled:
    https://console.cloud.google.com/apis/library/artifactregistry.googleapis.com?project=${PROJECT_ID}

  Docker repository exists:
    name:   ${AR_REPOSITORY}
    region: ${REGION}
    format: Docker
    url:    https://console.cloud.google.com/artifacts/docker/${PROJECT_ID}/${REGION}/${AR_REPOSITORY}?project=${PROJECT_ID}

  Your account can create/push images:
    roles/artifactregistry.admin or roles/artifactregistry.writer
EOF
  exit 0
fi

if ! gcloud artifacts repositories describe "${AR_REPOSITORY}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" >/dev/null 2>&1; then
  if [[ "${CREATE_REPO}" -eq 1 ]]; then
    gcloud artifacts repositories create "${AR_REPOSITORY}" \
      --project="${PROJECT_ID}" \
      --location="${REGION}" \
      --repository-format=docker \
      --description="verl RL training images"
  else
    echo "Artifact Registry repository not found: ${AR_REPOSITORY} in ${REGION}" >&2
    echo "Create it in Console or rerun without --no-create-repo." >&2
    exit 1
  fi
fi

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

docker build \
  --platform "${DOCKER_PLATFORM}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -f scripts/gcp/Dockerfile \
  -t "${IMAGE_URI}" \
  .

docker push "${IMAGE_URI}"

echo "Pushed ${IMAGE_URI}"
