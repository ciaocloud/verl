#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=load_env.sh
source "${SCRIPT_DIR}/load_env.sh"

usage() {
  cat <<'EOF'
Usage: scripts/gcp/status.sh [CUSTOM_JOB_ID_OR_NAME]

Without an argument, lists recent Vertex AI CustomJobs. With an argument,
describes that CustomJob.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

JOB_ID="${1:-}"

require_env PROJECT_ID
require_env REGION

if [[ -n "${JOB_ID}" ]]; then
  gcloud ai custom-jobs describe "${JOB_ID}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}"
else
  gcloud ai custom-jobs list \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --limit=10
fi
