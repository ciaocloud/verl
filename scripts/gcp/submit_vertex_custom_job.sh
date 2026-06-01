#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=load_env.sh
source "${SCRIPT_DIR}/load_env.sh"

usage() {
  cat <<'EOF'
Usage: scripts/gcp/submit_vertex_custom_job.sh [options]

Submits a Vertex AI CustomJob using the image and training settings from
scripts/gcp/.env or exported environment variables.

Options:
  --dry-run              Write the resolved job config and print the gcloud command.
  --config-out PATH      Write the generated Vertex jobSpec YAML to PATH.
  --job-name NAME        Override VERTEX_JOB_NAME for this submission.
  --image-uri URI        Override IMAGE_URI for this submission.
  -h, --help             Show this help.
EOF
}

DRY_RUN=0
CONFIG_OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --config-out)
      CONFIG_OUT="${2:?missing path after --config-out}"
      shift 2
      ;;
    --job-name)
      VERTEX_JOB_NAME="${2:?missing name after --job-name}"
      shift 2
      ;;
    --image-uri)
      IMAGE_URI="${2:?missing URI after --image-uri}"
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

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value core/project 2>/dev/null || true)}"
REGION="${REGION:-$(gcloud config get-value compute/region 2>/dev/null || true)}"
AR_REPOSITORY="${AR_REPOSITORY:-verl-rl}"
IMAGE_NAME="${IMAGE_NAME:-verl-gcp}"
IMAGE_TAG="${IMAGE_TAG:-$(git -C "$(repo_root)" rev-parse --short HEAD)}"
if [[ -z "${IMAGE_URI:-}" && -n "${PROJECT_ID}" && -n "${REGION}" ]]; then
  IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"
fi

require_env PROJECT_ID
require_env REGION
require_env IMAGE_URI
require_env TRAIN_FILES
require_env VAL_FILES
require_env GCS_CHECKPOINT_URI

VERTEX_JOB_NAME="${VERTEX_JOB_NAME:-verl-rl-$(date +%Y%m%d-%H%M%S)}"
TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-/workspace/verl/scripts/gcp/train.sh}"
if [[ -z "${TOTAL_TRAINING_STEPS+x}" && -z "${TOTAL_EPOCHS:-}" ]]; then
  RESOLVED_TOTAL_TRAINING_STEPS=2
else
  RESOLVED_TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
fi

if [[ -n "${CONFIG_OUT}" ]]; then
  CONFIG_FILE="${CONFIG_OUT}"
  mkdir -p "$(dirname "${CONFIG_FILE}")"
else
  CONFIG_FILE="$(mktemp -t verl-vertex-job.XXXXXX.yaml)"
fi

yaml_env() {
  local name="$1"
  local value="$2"
  [[ -n "${value}" ]] || return 0
  value="${value//\'/\'\'}"
  cat >>"${CONFIG_FILE}" <<YAML
    - name: ${name}
      value: '${value}'
YAML
}

cat >"${CONFIG_FILE}" <<YAML
workerPoolSpecs:
- machineSpec:
    machineType: ${VERTEX_MACHINE_TYPE:-g2-standard-16}
    acceleratorType: ${VERTEX_ACCELERATOR_TYPE:-NVIDIA_L4}
    acceleratorCount: ${VERTEX_ACCELERATOR_COUNT:-1}
  replicaCount: 1
  containerSpec:
    imageUri: ${IMAGE_URI}
    command:
    - /bin/bash
    - ${TRAIN_ENTRYPOINT}
    env:
YAML

yaml_env N_GPUS "${N_GPUS:-${VERTEX_ACCELERATOR_COUNT:-1}}"
yaml_env NVIDIA_VISIBLE_DEVICES "${NVIDIA_VISIBLE_DEVICES:-all}"
yaml_env NVIDIA_DRIVER_CAPABILITIES "${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
yaml_env LD_LIBRARY_PATH "${LD_LIBRARY_PATH:-/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/compat:/usr/local/cuda/lib64}"
yaml_env MODEL_PATH "${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
yaml_env CRITIC_MODEL_PATH "${CRITIC_MODEL_PATH:-${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}}"
yaml_env PROJECT_NAME "${PROJECT_NAME:-verl_gcp}"
yaml_env EXP_NAME "${EXP_NAME:-verl-run}"
yaml_env RL_ALGORITHM "${RL_ALGORITHM:-grpo}"
yaml_env ADV_ESTIMATOR "${ADV_ESTIMATOR:-}"
yaml_env TOTAL_TRAINING_STEPS "${RESOLVED_TOTAL_TRAINING_STEPS}"
yaml_env TOTAL_EPOCHS "${TOTAL_EPOCHS:-}"
yaml_env TRAIN_BATCH_SIZE "${TRAIN_BATCH_SIZE:-8}"
yaml_env DATALOADER_NUM_WORKERS "${DATALOADER_NUM_WORKERS:-0}"
yaml_env TRAIN_MAX_SAMPLES "${TRAIN_MAX_SAMPLES:--1}"
yaml_env VAL_MAX_SAMPLES "${VAL_MAX_SAMPLES:-16}"
yaml_env PPO_MINI_BATCH_SIZE "${PPO_MINI_BATCH_SIZE:-4}"
yaml_env PPO_MICRO_BATCH_SIZE_PER_GPU "${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
yaml_env ROLLOUT_N "${ROLLOUT_N:-2}"
yaml_env MAX_PROMPT_LENGTH "${MAX_PROMPT_LENGTH:-512}"
yaml_env MAX_RESPONSE_LENGTH "${MAX_RESPONSE_LENGTH:-512}"
yaml_env MAX_MODEL_LEN "${MAX_MODEL_LEN:-1024}"
yaml_env MAX_NUM_SEQS "${MAX_NUM_SEQS:-16}"
yaml_env MAX_NUM_BATCHED_TOKENS "${MAX_NUM_BATCHED_TOKENS:-8192}"
yaml_env ROLLOUT_AGENT_NUM_WORKERS "${ROLLOUT_AGENT_NUM_WORKERS:-1}"
yaml_env ROLLOUT_ENFORCE_EAGER "${ROLLOUT_ENFORCE_EAGER:-True}"
yaml_env ROLLOUT_ENABLE_CHUNKED_PREFILL "${ROLLOUT_ENABLE_CHUNKED_PREFILL:-False}"
yaml_env GPU_MEMORY_UTILIZATION "${GPU_MEMORY_UTILIZATION:-0.35}"
yaml_env ACTOR_LR "${ACTOR_LR:-1e-6}"
yaml_env CRITIC_LR "${CRITIC_LR:-1e-5}"
yaml_env CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU "${CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
yaml_env KL_COEF "${KL_COEF:-0.001}"
yaml_env KL_LOSS_COEF "${KL_LOSS_COEF:-0.001}"
yaml_env SAVE_FREQ "${SAVE_FREQ:-1}"
yaml_env TEST_FREQ "${TEST_FREQ:-1}"
yaml_env TRAINER_LOGGER "${TRAINER_LOGGER:-[\"console\"]}"
yaml_env GCS_OUTPUT_URI "${GCS_OUTPUT_URI:-}"
yaml_env GCS_CHECKPOINT_URI "${GCS_CHECKPOINT_URI}"
yaml_env CHECKPOINT_SYNC_INTERVAL_SECONDS "${CHECKPOINT_SYNC_INTERVAL_SECONDS:-300}"
yaml_env DATA_DIR "${DATA_DIR:-/workspace/data}"
yaml_env OUTPUT_DIR "${OUTPUT_DIR:-/workspace/outputs/${EXP_NAME:-verl-run}}"
yaml_env TRAIN_FILES "${TRAIN_FILES}"
yaml_env VAL_FILES "${VAL_FILES}"
yaml_env EXTRA_HYDRA_ARGS "${EXTRA_HYDRA_ARGS:-}"

cat >>"${CONFIG_FILE}" <<YAML
scheduling:
  timeout: ${VERTEX_TIMEOUT_SECONDS:-7200}s
YAML

if [[ -n "${VERTEX_SERVICE_ACCOUNT:-}" ]]; then
  cat >>"${CONFIG_FILE}" <<YAML
serviceAccount: ${VERTEX_SERVICE_ACCOUNT}
YAML
fi

if [[ -n "${VERTEX_OUTPUT_URI:-}" ]]; then
  cat >>"${CONFIG_FILE}" <<YAML
baseOutputDirectory:
  outputUriPrefix: ${VERTEX_OUTPUT_URI}
YAML
fi

echo "Submitting Vertex AI CustomJob ${VERTEX_JOB_NAME}"
echo "Config: ${CONFIG_FILE}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  cat <<EOF
Dry run only. Submit with:

gcloud ai custom-jobs create \\
  --project="${PROJECT_ID}" \\
  --region="${REGION}" \\
  --display-name="${VERTEX_JOB_NAME}" \\
  --config="${CONFIG_FILE}"
EOF
  exit 0
fi

gcloud ai custom-jobs create \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --display-name="${VERTEX_JOB_NAME}" \
  --config="${CONFIG_FILE}"

echo "Submitted ${VERTEX_JOB_NAME}"
echo "Console: https://console.cloud.google.com/vertex-ai/locations/${REGION}/training/custom-jobs?project=${PROJECT_ID}"
