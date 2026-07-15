#!/usr/bin/env bash
# Canonical YAML-first PLUTO paper benchmark entrypoint.

# Some conda activation hooks read optional unset variables, so nounset cannot
# be enabled before the shared runtime bootstrap.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Activate the requested runtime before choosing Python.  This avoids mixing a
# system resolver process with a conda training process on another host.
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/env_bootstrap.sh"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/python_runtime.sh"

exec "$PYTHON_BIN" "${REPO_ROOT}/scripts/experiments/run_lora_benchmark.py" "$@"
