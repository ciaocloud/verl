#!/usr/bin/env bash
# LOTIS training entrypoint for Vertex AI (Agent Platform) CustomJob.
#
# Standalone cloud runner. It owns only the cloud glue; ALL experiment config
# (algorithm, model, batch sizes, LOTIS knobs, epochs/steps, ...) is passed as
# raw Hydra overrides via the HYDRA_OVERRIDES env var in the jobspec. The glue:
#   - stage TRAIN_FILES/VAL_FILES from GCS to local disk,
#   - run the trainer in the FOREGROUND (so the container lives until it ends),
#   - sync checkpoints + tensorboard back to GCS periodically and on exit.
#
# bootstrap.sh execs this by default (TRAIN_ENTRYPOINT=lab/gcp/train.sh).
#
# Config env:   HYDRA_OVERRIDES — newline- or ';'-separated Hydra override lines
#               (e.g. "algorithm.adv_estimator=grpo;actor_rollout_ref.rollout.n=5").
#               '#'-prefixed and blank lines are ignored.
# Glue knobs (optional): EXP_NAME PROJ_NAME N_GPUS CHECKPOINT_SYNC_INTERVAL_SECONDS.
set -Eeuo pipefail

cd "${VERL_WORKDIR:-/workspace/verl}"

# Experiment name. If EXP_NAME is set explicitly it's used as-is; otherwise it's
# composed like the power scripts: <ALG>-<MODEL_SIZE>-<DATASET>-<MMDDHH>, e.g.
# LENRBF-0.5B-gsm8k-072714. Set ALG/MODEL_SIZE/DATASET in the jobspec.
if [[ -z "${DATE:-}" ]]; then
  DATE=$(date +%m%d%H)
fi
if [[ -z "${EXP_NAME:-}" ]]; then
  EXP_NAME="${ALG:-go}-${MODEL_SIZE:-0.5B}-${DATASET:-gsm8k}-${DATE}"
fi
PROJ_NAME="${PROJ_NAME:-verl-go}"
DATA_DIR="${DATA_DIR:-/workspace/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs/${EXP_NAME}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_DIR}/checkpoints}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_DIR}/tensorboard}"
export RAY_ADDRESS="${RAY_ADDRESS:-local}"


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

gcs_download_dir() { # gs://bucket/prefix  /local/dir   (no-op if prefix empty)
  python3 - "$1" "$2" <<'PY'
import sys
from pathlib import Path
from google.cloud import storage
uri, dst = sys.argv[1], Path(sys.argv[2])
bucket, _, prefix = uri[5:].partition("/")
prefix = prefix.rstrip("/")
client = storage.Client()
n = 0
for blob in client.list_blobs(bucket, prefix=prefix + "/" if prefix else None):
    if blob.name.endswith("/"):
        continue
    rel = blob.name[len(prefix):].lstrip("/") if prefix else blob.name
    out = dst / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(out))
    n += 1
print(f"staged {n} files from {uri} -> {dst}")
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

GCS_CHECKPOINT_URI="${GCS_CHECKPOINT_URI:-${GCS_BUCKET}/${EXP_NAME}}"
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
if [[ -z "${TRAIN_DATA:-}" ]]; then
  TRAIN_DATA="${GCS_BUCKET}/datasets/${DATASET}/train.parquet"
fi
if [[ -z "${VAL_DATA:-}" ]]; then
  VAL_DATA="${GCS_BUCKET}/datasets/${DATASET}/test.parquet"
fi
gcs_download "${TRAIN_DATA}" "${TRAIN_LOCAL}"
gcs_download "${VAL_DATA}" "${VAL_LOCAL}"

# ── Stage base model from GCS (optional) ──────────────────────────────────────
# If MODEL_GCS is set (or defaulted), pull the model dir from GCS to local disk
# and point actor_rollout_ref.model.path at it — avoids a HuggingFace download on
# every job. Set MODEL_GCS="" in the jobspec to disable and keep the HF hub path
# from HYDRA_OVERRIDES. Default matches the layout we upload models to:
#   gs://<bucket>/models/Qwen2.5-<MODEL_SIZE>-Instruct
MODEL_LOCAL=""
if [[ -z "${MODEL_GCS+x}" ]]; then
  MODEL_GCS="${GCS_BUCKET}/models/Qwen2.5-${MODEL_SIZE:-0.5B}-Instruct"
fi
if [[ -n "${MODEL_GCS}" ]]; then
  MODEL_LOCAL="/workspace/models/$(basename "${MODEL_GCS%/}")"
  gcs_download_dir "${MODEL_GCS}" "${MODEL_LOCAL}" || true
  # Only use the staged model if the download actually produced a HF config.
  if [[ ! -f "${MODEL_LOCAL}/config.json" ]]; then
    echo "WARN: no config.json under ${MODEL_LOCAL} after staging ${MODEL_GCS}; falling back to HYDRA_OVERRIDES model.path"
    MODEL_LOCAL=""
  fi
fi

# ── Stage prior checkpoints back down (preemption resume) ─────────────────────
# On a preempted/retried job the local disk is fresh, so pull any checkpoints we
# previously synced to GCS into CHECKPOINT_DIR. verl's resume_mode=auto then
# finds latest_checkpointed_iteration.txt + global_step_N/ and continues. This
# is a no-op on the first run (nothing in GCS yet). EXP_NAME must be stable
# across retries for the GCS path to match — pin it if the job may be preempted.
gcs_download_dir "${GCS_CHECKPOINT_URI%/}/checkpoints" "${CHECKPOINT_DIR}" || true

# ── Periodic checkpoint sync in background ────────────────────────────────────
sync_pid=""
if [[ "${CHECKPOINT_SYNC_INTERVAL_SECONDS:-300}" != "0" ]]; then
  ( while true; do sleep "${CHECKPOINT_SYNC_INTERVAL_SECONDS:-300}"; sync_to_gcs; done ) &
  sync_pid=$!
fi

export TENSORBOARD_DIR

# ── Trainer invocation ────────────────────────────────────────────────────────
# The full experiment config lives in the jobspec's HYDRA_OVERRIDES (one raw
# Hydra override per line). Each model/experiment gets its own self-contained
# jobspec YAML — edit it and resubmit; no git push, no image rebuild.
#
# In Hydra the LATER arg wins on a collision, so we append the runtime facts
# (staged data paths, real GPU count, run identity) AFTER HYDRA_OVERRIDES — they
# reflect container truth and must not be overridable by a stray jobspec line.
HYDRA_ARGS=()

# Experiment config from the jobspec: one override per line; skip blanks/comments.
if [[ -n "${HYDRA_OVERRIDES:-}" ]]; then
  while IFS= read -r arg; do
    arg="${arg#"${arg%%[![:space:]]*}"}"   # ltrim
    [[ -n "${arg}" ]] || continue          # skip blank lines
    [[ "${arg:0:1}" != "#" ]] || continue  # skip comment lines
    HYDRA_ARGS+=("${arg}")
  done <<< "${HYDRA_OVERRIDES}"
fi

# Runtime facts — appended last so they win over anything in HYDRA_OVERRIDES.
HYDRA_ARGS+=(
  custom_reward_function.path="${VERL_WORKDIR:-/workspace/verl}/lab/reward.py"
  data.train_files="${TRAIN_LOCAL}"
  data.val_files="${VAL_LOCAL}"
  trainer.n_gpus_per_node="${N_GPUS}"
  trainer.project_name="${PROJ_NAME}"
  trainer.experiment_name="${EXP_NAME}"
  trainer.default_local_dir="${CHECKPOINT_DIR}"
  trainer.resume_mode="${RESUME_MODE:-auto}"
)

# If a base model was staged from GCS, point the actor at the local copy (wins
# over any actor_rollout_ref.model.path in HYDRA_OVERRIDES).
if [[ -n "${MODEL_LOCAL}" ]]; then
  HYDRA_ARGS+=(actor_rollout_ref.model.path="${MODEL_LOCAL}")
fi

echo "Starting LOTIS training: exp=${EXP_NAME} n_gpus=${N_GPUS}"
printf '  %s\n' "${HYDRA_ARGS[@]}"
python3 -m verl.trainer.main_ppo "${HYDRA_ARGS[@]}" "$@"
