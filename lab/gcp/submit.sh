#!/usr/bin/env bash
# Submit a static Vertex AI (Agent Platform) CustomJob spec as-is.
#
# Everything (infra + params) lives in the YAML; this is a thin wrapper over
# `gcloud ai custom-jobs create`. To review a spec, just open the YAML.
#
# Usage:
#   lab/gcp/submit.sh lab/gcp/jobspec.lotis-rbf.yaml [options]
#
# Options:
#   --project ID         Override PROJECT_ID (default: gcloud config core/project).
#   --region REGION      Override REGION (default: gcloud config compute/region).
#   --display-name NAME  Job display name (default: jobspec file stem).
#   --dry-run            Print the gcloud command without submitting.
#   -h, --help           Show this help.
set -euo pipefail

usage() { sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

CONFIG_FILE=""
PROJECT_ID="${PROJECT_ID:-}"
REGION="${REGION:-}"
DISPLAY_NAME=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT_ID="${2:?missing id after --project}"; shift 2 ;;
    --region) REGION="${2:?missing region after --region}"; shift 2 ;;
    --display-name) DISPLAY_NAME="${2:?missing name after --display-name}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)
      if [[ -n "${CONFIG_FILE}" ]]; then
        echo "Unexpected extra argument: $1" >&2; usage >&2; exit 2
      fi
      CONFIG_FILE="$1"; shift ;;
  esac
done

if [[ -z "${CONFIG_FILE}" ]]; then
  echo "Missing jobspec YAML path." >&2; usage >&2; exit 2
fi
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Jobspec file not found: ${CONFIG_FILE}" >&2; exit 1
fi

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value core/project 2>/dev/null || true)}"
REGION="${REGION:-$(gcloud config get-value compute/region 2>/dev/null || true)}"
[[ -n "${PROJECT_ID}" ]] || { echo "PROJECT_ID empty. Pass --project or set gcloud config." >&2; exit 1; }
[[ -n "${REGION}" ]] || { echo "REGION empty. Pass --region or set gcloud config." >&2; exit 1; }

if [[ -z "${DISPLAY_NAME}" ]]; then
  DISPLAY_NAME="$(basename "${CONFIG_FILE}")"
  DISPLAY_NAME="${DISPLAY_NAME%.yaml}"
  DISPLAY_NAME="${DISPLAY_NAME%.yml}"
fi

echo "Vertex AI CustomJob"
echo "  PROJECT_ID:   ${PROJECT_ID}"
echo "  REGION:       ${REGION}"
echo "  DISPLAY_NAME: ${DISPLAY_NAME}"
echo "  CONFIG:       ${CONFIG_FILE}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  cat <<EOF

Dry run. Submit with:

gcloud ai custom-jobs create \\
  --project="${PROJECT_ID}" \\
  --region="${REGION}" \\
  --display-name="${DISPLAY_NAME}" \\
  --config="${CONFIG_FILE}"
EOF
  exit 0
fi

gcloud ai custom-jobs create \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --display-name="${DISPLAY_NAME}" \
  --config="${CONFIG_FILE}"

echo "Submitted ${DISPLAY_NAME}"
echo "Console: https://console.cloud.google.com/vertex-ai/locations/${REGION}/training/custom-jobs?project=${PROJECT_ID}"
