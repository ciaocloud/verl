#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=load_env.sh
source "${SCRIPT_DIR}/load_env.sh"
# shellcheck source=tensorboard_utils.sh
source "${SCRIPT_DIR}/tensorboard_utils.sh"

usage() {
  cat <<'EOF'
Usage: scripts/gcp/setup_tensorboard_iam.sh [options]

Creates or updates a least-privilege custom IAM role for live Vertex AI
TensorBoard uploads and grants it to the Vertex training service account.

The default mode is dry-run. Use --apply to change IAM.

Options:
  --apply              Create/update the role and grant it to the service account.
  --role-id ID         Custom role id. Defaults to vertexTensorboardUploader.
  --service-account SA Service account email. Defaults to VERTEX_SERVICE_ACCOUNT.
  -h, --help           Show this help.
EOF
}

APPLY=0
ROLE_ID="${TENSORBOARD_UPLOADER_ROLE_ID:-vertexTensorboardUploader}"
SERVICE_ACCOUNT="${VERTEX_SERVICE_ACCOUNT:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      APPLY=1
      shift
      ;;
    --role-id)
      ROLE_ID="${2:?missing id after --role-id}"
      shift 2
      ;;
    --service-account)
      SERVICE_ACCOUNT="${2:?missing email after --service-account}"
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

require_env PROJECT_ID
if [[ -z "${SERVICE_ACCOUNT}" ]]; then
  echo "Missing service account. Set VERTEX_SERVICE_ACCOUNT or pass --service-account." >&2
  exit 1
fi

ROLE_NAME="projects/${PROJECT_ID}/roles/${ROLE_ID}"
TITLE="${TENSORBOARD_UPLOADER_ROLE_TITLE:-Vertex TensorBoard Uploader}"
DESCRIPTION="${TENSORBOARD_UPLOADER_ROLE_DESCRIPTION:-Uploads live TensorBoard event data from Vertex training jobs.}"
PERMISSIONS_CSV="$(tensorboard_required_permissions | paste -sd, -)"

cat <<EOF
Resolved TensorBoard IAM setup
  PROJECT_ID:      ${PROJECT_ID}
  SERVICE_ACCOUNT: ${SERVICE_ACCOUNT}
  ROLE_ID:         ${ROLE_ID}
  ROLE_NAME:       ${ROLE_NAME}
  MODE:            $([[ "${APPLY}" -eq 1 ]] && echo apply || echo dry-run)

Permissions:
$(tensorboard_required_permissions | sed 's/^/  - /')
EOF

if [[ "${APPLY}" -ne 1 ]]; then
  cat <<EOF

Dry run only. To apply:

  scripts/gcp/setup_tensorboard_iam.sh --apply
EOF
  exit 0
fi

if gcloud iam roles describe "${ROLE_ID}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud iam roles update "${ROLE_ID}" \
    --project="${PROJECT_ID}" \
    --title="${TITLE}" \
    --description="${DESCRIPTION}" \
    --permissions="${PERMISSIONS_CSV}" \
    --stage=GA
else
  gcloud iam roles create "${ROLE_ID}" \
    --project="${PROJECT_ID}" \
    --title="${TITLE}" \
    --description="${DESCRIPTION}" \
    --permissions="${PERMISSIONS_CSV}" \
    --stage=GA
fi

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="${ROLE_NAME}" \
  --condition=None \
  --quiet

echo "Granted ${ROLE_NAME} to ${SERVICE_ACCOUNT}"
