#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=load_env.sh
source "${SCRIPT_DIR}/load_env.sh"
# shellcheck source=tensorboard_utils.sh
source "${SCRIPT_DIR}/tensorboard_utils.sh"

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
  --no-tensorboard    Do not create/resolve a managed Vertex TensorBoard.
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
ENABLE_TENSORBOARD="${ENABLE_VERTEX_TENSORBOARD:-1}"
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
    --no-tensorboard)
      ENABLE_TENSORBOARD=0
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

resolve_tensorboard_resource() {
  local can_create="$1"
  local display_name
  local resource_name

  [[ "${ENABLE_TENSORBOARD}" != "0" ]] || return 0
  require_env PROJECT_ID
  require_env REGION

  if [[ -n "${VERTEX_TENSORBOARD_RESOURCE_NAME:-}" ]]; then
    resource_name="$(gcloud ai tensorboards describe "${VERTEX_TENSORBOARD_RESOURCE_NAME}" \
      --project="${PROJECT_ID}" \
      --region="${REGION}" \
      --format='value(name)')"
    resource_name="${resource_name:-${VERTEX_TENSORBOARD_RESOURCE_NAME}}"
    VERTEX_TENSORBOARD_RESOURCE_NAME="$(tensorboard_resource_for_project_id "${PROJECT_ID}" "${resource_name}")"
    return 0
  fi

  display_name="${VERTEX_TENSORBOARD_DISPLAY_NAME:-verl-rl}"
  resource_name="$(gcloud ai tensorboards list \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --filter="displayName=${display_name}" \
    --format='value(name)' | head -n 1)"

  if [[ -z "${resource_name}" && "${can_create}" -eq 1 ]]; then
    echo "Creating Vertex AI TensorBoard ${display_name}"
    gcloud ai tensorboards create \
      --project="${PROJECT_ID}" \
      --region="${REGION}" \
      --display-name="${display_name}" \
      --description="Managed TensorBoard for verl RL training" \
      --quiet >/dev/null

    resource_name="$(gcloud ai tensorboards list \
      --project="${PROJECT_ID}" \
      --region="${REGION}" \
      --filter="displayName=${display_name}" \
      --format='value(name)' | head -n 1)"
  fi

  if [[ -z "${resource_name}" ]]; then
    echo "No Vertex AI TensorBoard named ${display_name} exists yet. Submit without --dry-run to create it, or set VERTEX_TENSORBOARD_RESOURCE_NAME." >&2
    return 0
  fi

  VERTEX_TENSORBOARD_RESOURCE_NAME="$(tensorboard_resource_for_project_id "${PROJECT_ID}" "${resource_name}")"
}

check_tensorboard_service_account_iam() {
  [[ "${ENABLE_TENSORBOARD}" != "0" ]] || return 0
  [[ -n "${VERTEX_TENSORBOARD_RESOURCE_NAME:-}" ]] || return 0
  [[ -n "${VERTEX_SERVICE_ACCOUNT:-}" ]] || return 0
  [[ "${SKIP_TENSORBOARD_IAM_CHECK:-0}" != "1" ]] || return 0

  local roles
  local role
  roles="$(gcloud projects get-iam-policy "${PROJECT_ID}" \
    --flatten='bindings[].members' \
    --filter="bindings.members:serviceAccount:${VERTEX_SERVICE_ACCOUNT}" \
    --format='value(bindings.role)')"

  if printf '%s\n' "${roles}" | grep -Eq '^(roles/aiplatform.user|roles/aiplatform.admin|roles/owner|roles/editor)$'; then
    return 0
  fi

  while IFS= read -r role; do
    [[ -n "${role}" ]] || continue
    if tensorboard_role_has_required_permissions "${PROJECT_ID}" "${role}"; then
      return 0
    fi
  done <<<"${roles}"

  cat >&2 <<EOF
The Vertex training service account does not have Vertex AI permissions needed
for live Managed TensorBoard uploads.

  service account: ${VERTEX_SERVICE_ACCOUNT}
  tensorboard:     ${VERTEX_TENSORBOARD_RESOURCE_NAME}

Grant a role with aiplatform.tensorboards.get and TensorBoard upload
permissions before submitting, for example:

  gcloud projects add-iam-policy-binding ${PROJECT_ID} \\
    --member=serviceAccount:${VERTEX_SERVICE_ACCOUNT} \\
    --role=roles/aiplatform.user

For least privilege, grant a custom role that includes the TensorBoard upload
permissions documented in scripts/gcp/README.md. Set SKIP_TENSORBOARD_IAM_CHECK=1
only if an equivalent permission path exists but cannot be detected by gcloud.
EOF
  exit 1
}

create_tensorboard_experiment() {
  [[ "${ENABLE_TENSORBOARD}" != "0" ]] || return 0
  [[ -n "${VERTEX_TENSORBOARD_RESOURCE_NAME:-}" ]] || return 0
  [[ -n "${VERTEX_TENSORBOARD_EXPERIMENT_NAME:-}" ]] || return 0

  local token
  local body
  local response_file
  local status
  local url

  token="$(gcloud auth print-access-token)"
  body="$(python3 -c 'import json, sys; print(json.dumps({"displayName": sys.argv[1]}))' "${VERTEX_TENSORBOARD_EXPERIMENT_NAME}")"
  response_file="$(mktemp -t vertex-tensorboard-experiment.XXXXXX.json)"
  url="https://${REGION}-aiplatform.googleapis.com/v1/${VERTEX_TENSORBOARD_RESOURCE_NAME}/experiments?tensorboardExperimentId=${VERTEX_TENSORBOARD_EXPERIMENT_NAME}"

  status="$(curl -sS -o "${response_file}" -w '%{http_code}' \
    -X POST \
    -H "Authorization: Bearer ${token}" \
    -H "Content-Type: application/json" \
    "${url}" \
    -d "${body}")"

  case "${status}" in
    200|201)
      echo "Created TensorBoard experiment ${VERTEX_TENSORBOARD_EXPERIMENT_NAME}"
      ;;
    409)
      echo "TensorBoard experiment already exists: ${VERTEX_TENSORBOARD_EXPERIMENT_NAME}"
      ;;
    *)
      echo "Failed to create TensorBoard experiment ${VERTEX_TENSORBOARD_EXPERIMENT_NAME} (HTTP ${status})" >&2
      cat "${response_file}" >&2
      rm -f "${response_file}"
      exit 1
      ;;
  esac
  rm -f "${response_file}"
}

if [[ "${REUSE_ENV_PATHS}" -eq 0 && -n "${GCS_BUCKET:-}" ]]; then
  GCS_CHECKPOINT_URI="gs://${GCS_BUCKET}/checkpoints/${EXP_NAME}"
  GCS_METRICS_URI="gs://${GCS_BUCKET}/metrics/${EXP_NAME}"
  GCS_OUTPUT_URI="gs://${GCS_BUCKET}/outputs/${EXP_NAME}"
  VERTEX_OUTPUT_URI="gs://${GCS_BUCKET}/vertex/${EXP_NAME}"
fi

CREATE_TENSORBOARD=0
if [[ "${DO_SUBMIT}" -eq 1 && "${DRY_RUN}" -eq 0 ]]; then
  CREATE_TENSORBOARD=1
fi
resolve_tensorboard_resource "${CREATE_TENSORBOARD}"

if [[ -n "${VERTEX_TENSORBOARD_RESOURCE_NAME:-}" ]]; then
  TRAINER_LOGGER="$(ensure_tensorboard_logger "${TRAINER_LOGGER:-}")"
  VERTEX_TENSORBOARD_EXPERIMENT_NAME="${VERTEX_TENSORBOARD_EXPERIMENT_NAME:-${EXP_NAME}}"
  VERTEX_TENSORBOARD_EXPERIMENT_URL="$(tensorboard_experiment_url \
    "${REGION}" \
    "${VERTEX_TENSORBOARD_RESOURCE_NAME}" \
    "${VERTEX_TENSORBOARD_EXPERIMENT_NAME}")"
fi

if [[ "${DO_SUBMIT}" -eq 1 && "${DRY_RUN}" -eq 0 ]]; then
  check_tensorboard_service_account_iam
  create_tensorboard_experiment
fi

export PROJECT_ID REGION AR_REPOSITORY IMAGE_NAME IMAGE_TAG IMAGE_URI
export RUN_ID EXP_NAME VERTEX_JOB_NAME
export GCS_CHECKPOINT_URI GCS_METRICS_URI GCS_OUTPUT_URI VERTEX_OUTPUT_URI
export TRAINER_LOGGER
export VERTEX_TENSORBOARD_RESOURCE_NAME VERTEX_TENSORBOARD_EXPERIMENT_NAME VERTEX_TENSORBOARD_EXPERIMENT_URL

cat <<EOF
Resolved Vertex experiment
  RUN_ID:             ${RUN_ID}
  EXP_NAME:           ${EXP_NAME}
  VERTEX_JOB_NAME:    ${VERTEX_JOB_NAME}
  IMAGE_URI:          ${IMAGE_URI}
  TRAIN_FILES:        ${TRAIN_FILES:-<missing>}
  VAL_FILES:          ${VAL_FILES:-<missing>}
  GCS_CHECKPOINT_URI: ${GCS_CHECKPOINT_URI:-<missing>}
  GCS_METRICS_URI:    ${GCS_METRICS_URI:-<empty>}
  GCS_OUTPUT_URI:     ${GCS_OUTPUT_URI:-<empty>}
  TRAINER_LOGGER:     ${TRAINER_LOGGER:-<default>}
  TENSORBOARD:        ${VERTEX_TENSORBOARD_RESOURCE_NAME:-<disabled>}
  TENSORBOARD_URL:    ${VERTEX_TENSORBOARD_EXPERIMENT_URL:-<unavailable>}
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
