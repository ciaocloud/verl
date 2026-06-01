#!/usr/bin/env bash
set -Eeuo pipefail

cd "${VERL_WORKDIR:-/workspace/verl}"

if [[ -z "${N_GPUS:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    N_GPUS="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    N_GPUS=1
  fi
fi

PROJECT_NAME="${PROJECT_NAME:-verl_gcp}"
EXP_NAME="${EXP_NAME:-verl-run}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
CRITIC_MODEL_PATH="${CRITIC_MODEL_PATH:-${MODEL_PATH}}"
DATA_DIR="${DATA_DIR:-/workspace/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs/${EXP_NAME}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_DIR}/checkpoints}"
RL_ALGORITHM="${RL_ALGORITHM:-grpo}"
ADV_ESTIMATOR_OVERRIDE="${ADV_ESTIMATOR:-}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-}"
if [[ -z "${TOTAL_TRAINING_STEPS+x}" && -z "${TOTAL_EPOCHS}" ]]; then
  TOTAL_TRAINING_STEPS=2
else
  TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
fi

if [[ -z "${TRAIN_FILES:-}" || -z "${VAL_FILES:-}" || -z "${GCS_CHECKPOINT_URI:-}" ]]; then
  cat >&2 <<'EOF'
Missing required training storage settings.

Set all of these before launching the job:
  TRAIN_FILES=gs://bucket/path/train.parquet
  VAL_FILES=gs://bucket/path/test.parquet
  GCS_CHECKPOINT_URI=gs://bucket/path/checkpoints/experiment-name
EOF
  exit 1
fi

mkdir -p "${DATA_DIR}" "${OUTPUT_DIR}" "${CHECKPOINT_DIR}" "${DATA_DIR}/input"

download_gcs_file() {
  local uri="$1"
  local dst="$2"

  if command -v gcloud >/dev/null 2>&1; then
    gcloud storage cp "${uri}" "${dst}" >&2
    return
  fi

  python3 - "${uri}" "${dst}" <<'PY'
import sys
from pathlib import Path
from google.cloud import storage

uri, dst = sys.argv[1], Path(sys.argv[2])
if not uri.startswith("gs://"):
    raise SystemExit(f"expected gs:// URI, got {uri}")

bucket_name, _, blob_name = uri[5:].partition("/")
dst.parent.mkdir(parents=True, exist_ok=True)
storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(str(dst))
print(f"downloaded {uri} -> {dst}", file=sys.stderr)
PY
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

stage_files() {
  local label="$1"
  local paths_csv="$2"
  local staged=()
  local idx=0
  local raw path basename dst joined

  IFS=',' read -r -a paths <<<"${paths_csv}"
  for raw in "${paths[@]}"; do
    path="$(trim "${raw}")"
    [[ -n "${path}" ]] || continue
    idx=$((idx + 1))

    if [[ "${path}" == gs://* ]]; then
      basename="${path##*/}"
      [[ -n "${basename}" ]] || basename="${label}_${idx}.parquet"
      dst="${DATA_DIR}/input/${label}_${idx}_${basename}"
      download_gcs_file "${path}" "${dst}"
      staged+=("${dst}")
    else
      if [[ ! -f "${path}" ]]; then
        echo "${label} file does not exist: ${path}" >&2
        exit 1
      fi
      staged+=("${path}")
    fi
  done

  if [[ "${#staged[@]}" -eq 0 ]]; then
    echo "No ${label} files were provided." >&2
    exit 1
  fi

  if [[ "${#staged[@]}" -eq 1 ]]; then
    printf '%s' "${staged[0]}"
    return
  fi

  joined="$(IFS=,; echo "${staged[*]}")"
  printf '[%s]' "${joined}"
}

upload_dir_to_gcs() {
  local src="$1"
  local dst_uri="$2"

  [[ -d "${src}" ]] || return 0

  if command -v gcloud >/dev/null 2>&1; then
    gcloud storage cp --recursive "${src}" "${dst_uri%/}/"
    return
  fi

  python3 - "${src}" "${dst_uri}" <<'PY'
import sys
from pathlib import Path
from google.cloud import storage

src = Path(sys.argv[1])
uri = sys.argv[2]
if not uri.startswith("gs://"):
    raise SystemExit(f"expected gs:// URI, got {uri}")

bucket_name, _, prefix = uri[5:].partition("/")
client = storage.Client()
bucket = client.bucket(bucket_name)

for path in src.rglob("*"):
    if path.is_file():
        rel = path.relative_to(src)
        blob_name = "/".join(part for part in [prefix.rstrip("/"), src.name, str(rel)] if part)
        bucket.blob(blob_name).upload_from_filename(str(path))
        print(f"uploaded gs://{bucket_name}/{blob_name}")
PY
}

sync_checkpoints() {
  echo "Syncing checkpoints to ${GCS_CHECKPOINT_URI}"
  upload_dir_to_gcs "${CHECKPOINT_DIR}" "${GCS_CHECKPOINT_URI}"
}

sync_on_exit() {
  local exit_code=$?
  if [[ -n "${checkpoint_sync_pid:-}" ]]; then
    kill "${checkpoint_sync_pid}" >/dev/null 2>&1 || true
  fi
  sync_checkpoints || true
  if [[ -n "${GCS_OUTPUT_URI:-}" ]]; then
    echo "Uploading outputs to ${GCS_OUTPUT_URI}"
    upload_dir_to_gcs "${OUTPUT_DIR}" "${GCS_OUTPUT_URI}"
  fi
  exit "${exit_code}"
}

case "${RL_ALGORITHM}" in
  grpo)
    ADV_ESTIMATOR="${ADV_ESTIMATOR_OVERRIDE:-grpo}"
    USE_KL_IN_REWARD="${USE_KL_IN_REWARD:-False}"
    ACTOR_USE_KL_LOSS="${ACTOR_USE_KL_LOSS:-True}"
    ROLLOUT_N="${ROLLOUT_N:-2}"
    ;;
  ppo)
    ADV_ESTIMATOR="${ADV_ESTIMATOR_OVERRIDE:-gae}"
    USE_KL_IN_REWARD="${USE_KL_IN_REWARD:-True}"
    ACTOR_USE_KL_LOSS="${ACTOR_USE_KL_LOSS:-False}"
    ROLLOUT_N="${ROLLOUT_N:-1}"
    ;;
  *)
    echo "Unsupported RL_ALGORITHM=${RL_ALGORITHM}. Expected grpo or ppo." >&2
    exit 2
    ;;
esac

trap sync_on_exit EXIT

echo "verl GCP training"
echo "RL_ALGORITHM=${RL_ALGORITHM}"
echo "ADV_ESTIMATOR=${ADV_ESTIMATOR}"
echo "N_GPUS=${N_GPUS}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "CRITIC_MODEL_PATH=${CRITIC_MODEL_PATH}"
echo "TRAIN_FILES=${TRAIN_FILES}"
echo "VAL_FILES=${VAL_FILES}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "GCS_CHECKPOINT_URI=${GCS_CHECKPOINT_URI}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"

echo "GPU preflight"
ls -l /dev/nvidia* 2>/dev/null || true
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found"
fi
python3 - <<'PY' || true
import os
import ctypes
try:
    for lib in ("libcuda.so.1", "libcudart.so.12"):
        try:
            ctypes.CDLL(lib)
            print(f"{lib}=loads")
        except Exception as exc:
            print(f"{lib}=fails: {exc}")
    import torch
    print(f"torch.cuda.is_available={torch.cuda.is_available()}")
    print(f"torch.cuda.device_count={torch.cuda.device_count()}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
except Exception as exc:
    print(f"torch cuda preflight failed: {exc}")
PY

TRAIN_FILE_LOCAL="$(stage_files train "${TRAIN_FILES}")"
VAL_FILE_LOCAL="$(stage_files val "${VAL_FILES}")"

echo "TRAIN_FILE_LOCAL=${TRAIN_FILE_LOCAL}"
echo "VAL_FILE_LOCAL=${VAL_FILE_LOCAL}"

checkpoint_sync_pid=""
if [[ "${CHECKPOINT_SYNC_INTERVAL_SECONDS:-300}" != "0" ]]; then
  (
    while true; do
      sleep "${CHECKPOINT_SYNC_INTERVAL_SECONDS:-300}"
      sync_checkpoints || true
    done
  ) &
  checkpoint_sync_pid=$!
fi

HYDRA_ARGS=(
  "algorithm.adv_estimator=${ADV_ESTIMATOR}"
  "algorithm.use_kl_in_reward=${USE_KL_IN_REWARD:-False}"
  "algorithm.kl_ctrl.kl_coef=${KL_COEF:-0.001}"
  "data.train_files=${TRAIN_FILE_LOCAL}"
  "data.val_files=${VAL_FILE_LOCAL}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE:-8}"
  "data.dataloader_num_workers=${DATALOADER_NUM_WORKERS:-0}"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH:-512}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH:-512}"
  "data.train_max_samples=${TRAIN_MAX_SAMPLES:--1}"
  "data.val_max_samples=${VAL_MAX_SAMPLES:-16}"
  "data.filter_overlong_prompts=True"
  "data.truncation=error"
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.model.enable_gradient_checkpointing=${ENABLE_GRADIENT_CHECKPOINTING:-False}"
  "actor_rollout_ref.actor.optim.lr=${ACTOR_LR:-1e-6}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-4}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
  "actor_rollout_ref.actor.use_kl_loss=${ACTOR_USE_KL_LOSS:-True}"
  "actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF:-0.001}"
  "actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE:-low_var_kl}"
  "actor_rollout_ref.actor.entropy_coeff=${ACTOR_ENTROPY_COEFF:-0}"
  "actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD:-False}"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD:-False}"
  "actor_rollout_ref.rollout.name=${ROLLOUT_BACKEND:-vllm}"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE:-1}"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.35}"
  "actor_rollout_ref.rollout.n=${ROLLOUT_N:-2}"
  "actor_rollout_ref.rollout.agent.num_workers=${ROLLOUT_AGENT_NUM_WORKERS:-1}"
  "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN:-1024}"
  "actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS:-16}"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-8192}"
  "actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER:-True}"
  "actor_rollout_ref.rollout.enable_chunked_prefill=${ROLLOUT_ENABLE_CHUNKED_PREFILL:-False}"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
  "critic.optim.lr=${CRITIC_LR:-1e-5}"
  "critic.model.path=${CRITIC_MODEL_PATH}"
  "critic.ppo_micro_batch_size_per_gpu=${CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
  "trainer.critic_warmup=${CRITIC_WARMUP:-0}"
  "trainer.logger=${TRAINER_LOGGER:-[\"console\"]}"
  "trainer.val_before_train=${VAL_BEFORE_TRAIN:-False}"
  "trainer.n_gpus_per_node=${N_GPUS}"
  "trainer.nnodes=${NNODES:-1}"
  "trainer.save_freq=${SAVE_FREQ:-1}"
  "trainer.test_freq=${TEST_FREQ:-1}"
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.experiment_name=${EXP_NAME}"
  "trainer.default_local_dir=${CHECKPOINT_DIR}"
)

if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
  HYDRA_ARGS+=("trainer.total_training_steps=${TOTAL_TRAINING_STEPS}")
fi

if [[ -n "${TOTAL_EPOCHS}" ]]; then
  HYDRA_ARGS+=("trainer.total_epochs=${TOTAL_EPOCHS}")
fi

if [[ -n "${EXTRA_HYDRA_ARGS:-}" ]]; then
  while IFS= read -r raw_arg; do
    arg="$(trim "${raw_arg}")"
    [[ -n "${arg}" ]] || continue
    HYDRA_ARGS+=("${arg}")
  done < <(printf '%s\n' "${EXTRA_HYDRA_ARGS}" | tr ';' '\n')
fi

python3 -m verl.trainer.main_ppo "${HYDRA_ARGS[@]}" "$@"
