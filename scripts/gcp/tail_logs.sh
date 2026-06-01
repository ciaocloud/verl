#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=load_env.sh
source "${SCRIPT_DIR}/load_env.sh"

usage() {
  cat <<'EOF'
Usage: scripts/gcp/tail_logs.sh CUSTOM_JOB_ID_OR_NAME

Streams logs for a Vertex AI CustomJob.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

JOB_ID="${1:-}"

require_env PROJECT_ID
require_env REGION

if [[ -z "${JOB_ID}" ]]; then
  usage >&2
  exit 2
fi

gcloud ai custom-jobs stream-logs "${JOB_ID}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}"
