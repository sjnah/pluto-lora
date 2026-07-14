#!/usr/bin/env bash
# Backward-compatible entrypoint for bucketed percentile-EHU methods.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD="${METHOD:-llm}"

# shellcheck source=run_lora_experiment.sh
source "${SCRIPT_DIR}/run_lora_experiment.sh"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
