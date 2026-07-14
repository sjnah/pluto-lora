#!/usr/bin/env bash
# Backward-compatible entrypoint for the Uniform arm of the common protocol.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD=uniform
CURRICULUM_VERSION="${UNIFORM_CURRICULUM_VERSION:-${CURRICULUM_VERSION:-unversioned}}"

# shellcheck source=run_lora_experiment.sh
source "${SCRIPT_DIR}/run_lora_experiment.sh"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
