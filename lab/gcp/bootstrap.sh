#!/usr/bin/env bash
# Container entrypoint for the lab/gcp runner image.
#
# Baked into the image (see lab/gcp/Dockerfile). The image already contains a
# git checkout of the fork (cloned at build time), so this script only UPDATES
# it: fetch from origin, then hard-checkout the requested ref. Then it re-runs
# the editable install and hands off to the (now up-to-date) train.sh. This is
# what lets code changes ship via `git push` alone, with no image rebuild.
#
# Env:
#   GIT_REMOTE  Repo URL. Default: origin baked at build time, else the fork URL.
#   GIT_REF     Optional. Branch, tag, or commit SHA to run. If unset, the branch
#               baked into the image (default branch, e.g. main) is updated to its
#               remote tip. Set it (e.g. GIT_REF=lotis, or a commit SHA for a
#               reproducible run) to run a different ref.
#   REPO_DIR    Baked checkout location. Default: /workspace/verl.
#   TRAIN_ENTRYPOINT  Repo-relative script to exec. Default: lab/gcp/train.sh.
#                     e.g. lab/lotis/run_0.5b_lab.sh or scripts/gcp/train.sh.
#   SKIP_PIP_INSTALL  Set to 1 to skip `pip install -e` (faster start; safe when
#                     no packaging metadata changed).
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-/workspace/verl}"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  echo "bootstrap: no git checkout at ${REPO_DIR}; image may be built wrong." >&2
  exit 1
fi

# Allow overriding the remote (e.g. to swap to an authed URL for a private fork).
if [[ -n "${GIT_REMOTE:-}" ]]; then
  git -C "${REPO_DIR}" remote set-url origin "${GIT_REMOTE}"
fi

if [[ -n "${GIT_REF:-}" ]]; then
  # A specific ref was requested: fetch it and hard-checkout. Handles branches,
  # tags, and full commit SHAs. --tags keeps tag refs current.
  echo "bootstrap: updating ${REPO_DIR} to ${GIT_REF}"
  git -C "${REPO_DIR}" fetch --prune --tags origin "${GIT_REF}" || \
    git -C "${REPO_DIR}" fetch --prune --tags origin
  git -C "${REPO_DIR}" checkout -f "${GIT_REF}"
  # If GIT_REF is a branch, align it to the just-fetched remote tip.
  git -C "${REPO_DIR}" reset --hard "@{upstream}" 2>/dev/null || \
    git -C "${REPO_DIR}" reset --hard FETCH_HEAD 2>/dev/null || true
else
  # No ref requested: update the branch baked at build time to its remote tip.
  echo "bootstrap: no GIT_REF set; updating baked branch to origin tip"
  git -C "${REPO_DIR}" fetch --prune --tags origin
  git -C "${REPO_DIR}" reset --hard "@{upstream}" 2>/dev/null || true
fi

echo "bootstrap: running commit $(git -C "${REPO_DIR}" log -1 --oneline)"

# Refresh the editable install in case packaging metadata changed. Cheap:
# --no-deps, and deps are already baked into the image.
if [[ "${SKIP_PIP_INSTALL:-0}" != "1" ]]; then
  python3 -m pip install --no-deps -e "${REPO_DIR}" >/dev/null
fi

cd "${REPO_DIR}"
# Entrypoint script is selectable per-job (repo-relative path). No image rebuild
# needed to switch it, e.g. TRAIN_ENTRYPOINT=lab/lotis/run_0.5b_lab.sh.
TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-lab/gcp/train.sh}"
exec bash "${REPO_DIR}/${TRAIN_ENTRYPOINT}" "$@"
