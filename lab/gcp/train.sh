#!/usr/bin/env bash
# LOTIS training entrypoint for Vertex AI (Agent Platform) CustomJob.
#
# Standalone cloud runner. Mirrors the trainer invocation in
# lab/lotis/run_0.5b_lab.sh, plus the cloud glue that script omits:
#   - stage TRAIN_FILES/VAL_FILES from GCS to local disk,
#   - run the trainer in the FOREGROUND (so the container lives until it ends),
#   - sync checkpoints + tensorboard back to GCS periodically and on exit.
#
# bootstrap.sh execs this by default (TRAIN_ENTRYPOINT=lab/gcp/train.sh).
#
# Required env: TRAIN_FILES, VAL_FILES, GCS_CHECKPOINT_URI (all gs:// URIs).
# Common knobs (optional, with defaults): EXP_NAME PROJ_NAME MODEL_PATH
#   TOTAL_EPOCHS N_GPUS TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE ROLLOUT_N
#   GPU_MEMORY_UTILIZATION MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH SAVE_FREQ
#   TEST_FREQ LOTIS_LENGTH_ENABLE LOTIS_LENGTH_LR LOTIS_TOKEN_ENABLE
#   LOTIS_TOKEN_LR CHECKPOINT_SYNC_INTERVAL_SECONDS EXTRA_HYDRA_ARGS.
set -Eeuo pipefail

cd "${VERL_WORKDIR:-/workspace/verl}"

EXP_NAME="${EXP_NAME:-lotis-run}"
PROJ_NAME="${PROJ_NAME:-verl-lotis}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
DATA_DIR="${DATA_DIR:-/workspace/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs/${EXP_NAME}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_DIR}/checkpoints}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_DIR}/tensorboard}"
export RAY_ADDRESS="${RAY_ADDRESS:-local}"

if [[ -z "${TRAIN_FILES:-}" || -z "${VAL_FILES:-}" || -z "${GCS_CHECKPOINT_URI:-}" ]]; then
  echo "Missing required env: TRAIN_FILES, VAL_FILES, GCS_CHECKPOINT_URI (all gs:// URIs)." >&2
  exit 1
fi

if [[ -z "${N_GPUS:-}" ]]; then
  N_GPUS="$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L | wc -l | tr -d ' ' || echo 1)"
fi

mkdir -p "${DATA_DIR}" "${CHECKPOINT_DIR}" "${TENSORBOARD_DIR}"

# ── GCS helpers (google-cloud-storage is baked into the image) ────────────────
gcs_download() { # gs://bucket/obj  /local/path
  python3 - "$1" "$2" <<'PY'
import sys
from pathlib import Path
from google.cloud import storage
uri, dst = sys.argv[1], Path(sys.argv[2])
bucket, _, blob = uri[5:].partition("/")
dst.parent.mkdir(parents=True, exist_ok=True)
storage.Client().bucket(bucket).blob(blob).download_to_filename(str(dst))
print(f"staged {uri} -> {dst}")
PY
}

gcs_upload_dir() { # /local/dir  gs://bucket/prefix
  [[ -d "$1" ]] || return 0
  python3 - "$1" "$2" <<'PY'
import sys
from pathlib import Path
from google.cloud import storage
src, uri = Path(sys.argv[1]), sys.argv[2]
bucket, _, prefix = uri[5:].partition("/")
b = storage.Client().bucket(bucket)
for p in src.rglob("*"):
    if p.is_file():
        name = "/".join(x for x in [prefix.rstrip("/"), str(p.relative_to(src))] if x)
        b.blob(name).upload_from_filename(str(p))
PY
}

sync_to_gcs() {
  echo "Syncing checkpoints + tensorboard to ${GCS_CHECKPOINT_URI}"
  gcs_upload_dir "${CHECKPOINT_DIR}" "${GCS_CHECKPOINT_URI%/}/checkpoints" || true
  gcs_upload_dir "${TENSORBOARD_DIR}" "${GCS_CHECKPOINT_URI%/}/tensorboard" || true
}

on_exit() {
  local code=$?
  [[ -n "${sync_pid:-}" ]] && kill "${sync_pid}" >/dev/null 2>&1 || true
  sync_to_gcs
  exit "${code}"
}
trap on_exit EXIT

# ── Stage data from GCS ───────────────────────────────────────────────────────
TRAIN_LOCAL="${DATA_DIR}/train.parquet"
VAL_LOCAL="${DATA_DIR}/val.parquet"
gcs_download "${TRAIN_FILES}" "${TRAIN_LOCAL}"
gcs_download "${VAL_FILES}" "${VAL_LOCAL}"

# ── Periodic checkpoint sync in background ────────────────────────────────────
sync_pid=""
if [[ "${CHECKPOINT_SYNC_INTERVAL_SECONDS:-300}" != "0" ]]; then
  ( while true; do sleep "${CHECKPOINT_SYNC_INTERVAL_SECONDS:-300}"; sync_to_gcs; done ) &
  sync_pid=$!
fi

export TENSORBOARD_DIR

# ── Trainer invocation (mirrors lab/lotis/run_0.5b_lab.sh; foreground) ─────────
HYDRA_ARGS=(
  custom_reward_function.path="${VERL_WORKDIR:-/workspace/verl}/lab/reward.py"
  custom_reward_function.name=compute_score
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  data.train_files="${TRAIN_LOCAL}"
  data.val_files="${VAL_LOCAL}"
  data.train_batch_size="${TRAIN_BATCH_SIZE:-256}"
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-512}"
  data.max_response_length="${MAX_RESPONSE_LENGTH:-1024}"
  data.filter_overlong_prompts=True
  data.truncation=error
  actor_rollout_ref.model.path="${MODEL_PATH}"
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-6}"
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-64}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.actor.lotis.length_weight.enable="${LOTIS_LENGTH_ENABLE:-True}"
  actor_rollout_ref.actor.lotis.length_weight.lr="${LOTIS_LENGTH_LR:-0.01}"
  actor_rollout_ref.actor.lotis.token_weight.enable="${LOTIS_TOKEN_ENABLE:-False}"
  actor_rollout_ref.actor.lotis.token_weight.lr="${LOTIS_TOKEN_LR:-0.001}"
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.35}"
  actor_rollout_ref.rollout.n="${ROLLOUT_N:-5}"
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.top_k=-1
  actor_rollout_ref.rollout.val_kwargs.n=1
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0
  actor_rollout_ref.rollout.val_kwargs.top_p=0.7
  actor_rollout_ref.rollout.val_kwargs.top_k=-1
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
  trainer.logger=[console,tensorboard]
  trainer.log_val_generations=1
  trainer.val_before_train="${VAL_BEFORE_TRAIN:-False}"
  trainer.n_gpus_per_node="${N_GPUS}"
  trainer.nnodes=1
  trainer.save_freq="${SAVE_FREQ:-10}"
  trainer.test_freq="${TEST_FREQ:-10}"
  trainer.project_name="${PROJ_NAME}"
  trainer.experiment_name="${EXP_NAME}"
  trainer.default_local_dir="${CHECKPOINT_DIR}"
)

if [[ -n "${TOTAL_TRAINING_STEPS:-}" ]]; then
  HYDRA_ARGS+=(trainer.total_training_steps="${TOTAL_TRAINING_STEPS}")
else
  HYDRA_ARGS+=(trainer.total_epochs="${TOTAL_EPOCHS:-3}")
fi

if [[ -n "${EXTRA_HYDRA_ARGS:-}" ]]; then
  while IFS= read -r arg; do
    [[ -n "${arg// }" ]] && HYDRA_ARGS+=("${arg}")
  done < <(printf '%s\n' "${EXTRA_HYDRA_ARGS}" | tr ';' '\n')
fi

echo "Starting LOTIS training: exp=${EXP_NAME} model=${MODEL_PATH} n_gpus=${N_GPUS}"
python3 -m verl.trainer.main_ppo "${HYDRA_ARGS[@]}" "$@"
