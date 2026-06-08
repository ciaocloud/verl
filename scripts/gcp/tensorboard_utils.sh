#!/usr/bin/env bash

tensorboard_resource_to_url_path() {
  local resource_name="$1"
  printf '%s' "${resource_name//\//+}"
}

tensorboard_experiment_url() {
  local region="$1"
  local resource_name="$2"
  local experiment_name="$3"

  [[ -n "${region}" && -n "${resource_name}" && -n "${experiment_name}" ]] || return 0
  printf 'https://%s.tensorboard.googleusercontent.com/experiment/%s+experiments+%s' \
    "${region}" \
    "$(tensorboard_resource_to_url_path "${resource_name}")" \
    "${experiment_name}"
}

tensorboard_resource_for_project_id() {
  local project_id="$1"
  local resource_name="$2"

  if [[ "${resource_name}" =~ ^projects/[^/]+/locations/([^/]+)/tensorboards/([^/]+)$ ]]; then
    printf 'projects/%s/locations/%s/tensorboards/%s' "${project_id}" "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
    return 0
  fi

  printf '%s' "${resource_name}"
}

logger_includes_tensorboard() {
  [[ "${1:-}" == *tensorboard* ]]
}

ensure_tensorboard_logger() {
  local logger="${1:-}"
  if logger_includes_tensorboard "${logger}"; then
    printf '%s' "${logger}"
    return 0
  fi

  printf '%s' '["console","tensorboard","file"]'
}

tensorboard_required_permissions() {
  cat <<'EOF'
aiplatform.tensorboards.get
aiplatform.tensorboards.list
aiplatform.tensorboards.recordAccess
aiplatform.tensorboardExperiments.create
aiplatform.tensorboardExperiments.get
aiplatform.tensorboardExperiments.list
aiplatform.tensorboardExperiments.update
aiplatform.tensorboardExperiments.write
aiplatform.tensorboardRuns.batchCreate
aiplatform.tensorboardRuns.create
aiplatform.tensorboardRuns.get
aiplatform.tensorboardRuns.list
aiplatform.tensorboardRuns.update
aiplatform.tensorboardRuns.write
aiplatform.tensorboardTimeSeries.batchCreate
aiplatform.tensorboardTimeSeries.batchRead
aiplatform.tensorboardTimeSeries.create
aiplatform.tensorboardTimeSeries.get
aiplatform.tensorboardTimeSeries.list
aiplatform.tensorboardTimeSeries.read
aiplatform.tensorboardTimeSeries.update
aiplatform.metadataStores.create
aiplatform.metadataStores.get
aiplatform.artifacts.create
aiplatform.artifacts.get
aiplatform.artifacts.list
aiplatform.artifacts.update
aiplatform.contexts.addContextArtifactsAndExecutions
aiplatform.contexts.addContextChildren
aiplatform.contexts.create
aiplatform.contexts.get
aiplatform.contexts.list
aiplatform.contexts.queryContextLineageSubgraph
aiplatform.contexts.update
aiplatform.executions.addExecutionEvents
aiplatform.executions.create
aiplatform.executions.get
aiplatform.executions.list
aiplatform.executions.queryExecutionInputsAndOutputs
aiplatform.executions.update
aiplatform.metadataStores.list
resourcemanager.projects.get
EOF
}

tensorboard_role_has_required_permissions() {
  local project_id="$1"
  local role="$2"
  local role_project="${project_id}"
  local role_id="${role}"
  local permissions
  local permission

  if [[ "${role}" =~ ^projects/([^/]+)/roles/([^/]+)$ ]]; then
    role_project="${BASH_REMATCH[1]}"
    role_id="${BASH_REMATCH[2]}"
  fi

  permissions="$(gcloud iam roles describe "${role_id}" \
    --project="${role_project}" \
    --format='value(includedPermissions)' 2>/dev/null | tr ';' '\n' || true)"
  [[ -n "${permissions}" ]] || return 1

  while IFS= read -r permission; do
    [[ -n "${permission}" ]] || continue
    if ! printf '%s\n' "${permissions}" | grep -Fxq "${permission}"; then
      return 1
    fi
  done < <(tensorboard_required_permissions)
  return 0
}
