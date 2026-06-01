#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=load_env.sh
source "${SCRIPT_DIR}/load_env.sh"

usage() {
  cat <<'EOF'
Usage: scripts/gcp/launch_vertex_experiment.sh [options]

Production launcher for one Vertex AI experiment run. It creates a run id,
optionally builds and pushes an immutable image tag, resolves per-run GCS
checkpoint/output paths, then submits or dry-runs the Vertex CustomJob.

Options:
  --build             Build and push a new image tag before submitting.
  --no-submit         Resolve config and optionally build, but do not submit.
  --dry-run           Generate the Vertex jobSpec YAML without submitting.
  --run-id ID         Override RUN_ID. Defaults to timestamp + git SHA.
  --exp-name NAME     Override the base experiment name.
  --image-tag TAG     Override IMAGE_TAG. With --build, defaults to RUN_ID.
  --reuse-env-paths   Keep GCS_CHECKPOINT_URI/GCS_OUTPUT_URI/VERTEX_OUTPUT_URI from env.
  --config-out PATH   Write the generated Vertex jobSpec YAML to PATH.
  -h, --help          Show this help.

Examples:
  scripts/gcp/launch_vertex_experiment.sh --build
  scripts/gcp/launch_vertex_experiment.sh --dry-run --config-out /tmp/job.yaml
EOF
}

DO_BUILD=0
DO_SUBMIT=1
DRY_RUN=0
REUSE_ENV_PATHS=0
CONFIG_OUT=""
RUN_ID="${RUN_ID:-}"
BASE_EXP_NAME=""
REQUESTED_IMAGE_TAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)
      DO_BUILD=1
      shift
      ;;
    --no-submit)
      DO_SUBMIT=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --run-id)
      RUN_ID="${2:?missing id after --run-id}"
      shift 2
      ;;
    --exp-name)
      BASE_EXP_NAME="${2:?missing name after --exp-name}"
      shift 2
      ;;
    --image-tag)
      REQUESTED_IMAGE_TAG="${2:?missing tag after --image-tag}"
      shift 2
      ;;
    --reuse-env-paths)
      REUSE_ENV_PATHS=1
      shift
      ;;
    --config-out)
      CONFIG_OUT="${2:?missing path after --config-out}"
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

GIT_SHA="$(git -C "$(repo_root)" rev-parse --short HEAD 2>/dev/null || echo nogit)"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-${GIT_SHA}}"
BASE_EXP_NAME="${BASE_EXP_NAME:-${EXP_NAME:-grpo}}"
EXP_NAME="${BASE_EXP_NAME}-${RUN_ID}"
BASE_VERTEX_JOB_NAME="${VERTEX_JOB_NAME:-${BASE_EXP_NAME}}"
VERTEX_JOB_NAME="${BASE_VERTEX_JOB_NAME}-${RUN_ID}"

if [[ -n "${REQUESTED_IMAGE_TAG}" ]]; then
  IMAGE_TAG="${REQUESTED_IMAGE_TAG}"
elif [[ "${DO_BUILD}" -eq 1 ]]; then
  IMAGE_TAG="${RUN_ID}"
else
  IMAGE_TAG="${IMAGE_TAG:-runner}"
fi

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value core/project 2>/dev/null || true)}"
REGION="${REGION:-$(gcloud config get-value compute/region 2>/dev/null || true)}"
AR_REPOSITORY="${AR_REPOSITORY:-verl-rl}"
IMAGE_NAME="${IMAGE_NAME:-verl-gcp}"
IMAGE_URI="${IMAGE_URI:-${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}}"

if [[ "${REUSE_ENV_PATHS}" -eq 0 && -n "${GCS_BUCKET:-}" ]]; then
  GCS_CHECKPOINT_URI="gs://${GCS_BUCKET}/checkpoints/${EXP_NAME}"
  GCS_OUTPUT_URI="gs://${GCS_BUCKET}/outputs/${EXP_NAME}"
  VERTEX_OUTPUT_URI="gs://${GCS_BUCKET}/vertex/${EXP_NAME}"
fi

export PROJECT_ID REGION AR_REPOSITORY IMAGE_NAME IMAGE_TAG IMAGE_URI
export RUN_ID EXP_NAME VERTEX_JOB_NAME
export GCS_CHECKPOINT_URI GCS_OUTPUT_URI VERTEX_OUTPUT_URI

cat <<EOF
Resolved Vertex experiment
  RUN_ID:             ${RUN_ID}
  EXP_NAME:           ${EXP_NAME}
  VERTEX_JOB_NAME:    ${VERTEX_JOB_NAME}
  IMAGE_URI:          ${IMAGE_URI}
  TRAIN_FILES:        ${TRAIN_FILES:-<missing>}
  VAL_FILES:          ${VAL_FILES:-<missing>}
  GCS_CHECKPOINT_URI: ${GCS_CHECKPOINT_URI:-<missing>}
  GCS_OUTPUT_URI:     ${GCS_OUTPUT_URI:-<empty>}
EOF

if [[ "${DO_BUILD}" -eq 1 ]]; then
  ENV_FILE=/dev/null "${SCRIPT_DIR}/build_and_push.sh" --tag "${IMAGE_TAG}"
fi

SUBMIT_ARGS=()
if [[ -n "${CONFIG_OUT}" ]]; then
  SUBMIT_ARGS+=(--config-out "${CONFIG_OUT}")
fi
if [[ "${DRY_RUN}" -eq 1 || "${DO_SUBMIT}" -eq 0 ]]; then
  SUBMIT_ARGS+=(--dry-run)
fi

if [[ "${#SUBMIT_ARGS[@]}" -gt 0 ]]; then
  ENV_FILE=/dev/null "${SCRIPT_DIR}/submit_vertex_custom_job.sh" "${SUBMIT_ARGS[@]}"
else
  ENV_FILE=/dev/null "${SCRIPT_DIR}/submit_vertex_custom_job.sh"
fi
