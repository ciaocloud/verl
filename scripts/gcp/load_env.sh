#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env}"

if [[ -f "${ENV_FILE}" ]]; then
  __env_override_names=()
  __env_override_values=()
  while IFS= read -r __env_line || [[ -n "${__env_line}" ]]; do
    __env_line="${__env_line#"${__env_line%%[![:space:]]*}"}"
    [[ -n "${__env_line}" && "${__env_line}" != \#* ]] || continue
    __env_line="${__env_line#export }"
    __env_name="${__env_line%%=*}"
    [[ "${__env_name}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [[ -n "${!__env_name+x}" ]]; then
      __env_override_names+=("${__env_name}")
      __env_override_values+=("${!__env_name}")
    fi
  done <"${ENV_FILE}"

  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a

  for __env_idx in "${!__env_override_names[@]}"; do
    __env_name="${__env_override_names[${__env_idx}]}"
    printf -v "${__env_name}" '%s' "${__env_override_values[${__env_idx}]}"
    export "${__env_name}"
  done
  unset __env_idx __env_line __env_name __env_override_names __env_override_values
fi

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    echo "Set it in ${ENV_FILE} or export it before running this script." >&2
    exit 1
  fi
}

repo_root() {
  cd "${SCRIPT_DIR}/../.." >/dev/null
  pwd
}
