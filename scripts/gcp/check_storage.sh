#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=load_env.sh
source "${SCRIPT_DIR}/load_env.sh"

require_env PROJECT_ID
require_env TRAIN_FILES
require_env VAL_FILES
require_env GCS_CHECKPOINT_URI

check_gcs_object() {
  local label="$1"
  local uri="$2"
  local raw path

  IFS=',' read -r -a paths <<<"${uri}"
  for raw in "${paths[@]}"; do
    path="${raw#"${raw%%[![:space:]]*}"}"
    path="${path%"${path##*[![:space:]]}"}"
    [[ -n "${path}" ]] || continue

    if [[ "${path}" != gs://* ]]; then
      echo "${label}: local path configured, skipping GCS check: ${path}"
      continue
    fi

    echo "Checking ${label}: ${path}"
    gcloud storage ls "${path}" --project="${PROJECT_ID}" >/dev/null
  done
}

check_gcs_prefix() {
  local label="$1"
  local uri="$2"

  if [[ "${uri}" != gs://* ]]; then
    echo "${label} must be a gs:// URI: ${uri}" >&2
    exit 1
  fi

  local bucket="${uri#gs://}"
  bucket="${bucket%%/*}"
  echo "Checking ${label} bucket: gs://${bucket}"
  gcloud storage buckets describe "gs://${bucket}" --project="${PROJECT_ID}" >/dev/null
}

check_gcs_object "TRAIN_FILES" "${TRAIN_FILES}"
check_gcs_object "VAL_FILES" "${VAL_FILES}"
check_gcs_prefix "GCS_CHECKPOINT_URI" "${GCS_CHECKPOINT_URI}"

if [[ -n "${GCS_OUTPUT_URI:-}" ]]; then
  check_gcs_prefix "GCS_OUTPUT_URI" "${GCS_OUTPUT_URI}"
fi

if [[ -n "${VERTEX_OUTPUT_URI:-}" ]]; then
  check_gcs_prefix "VERTEX_OUTPUT_URI" "${VERTEX_OUTPUT_URI}"
fi

echo "Storage preflight passed."
