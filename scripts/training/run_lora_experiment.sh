#!/bin/bash
# Compatibility wrapper for the split LoRA experiment launchers.
#
# Prefer calling the explicit script directly:
#   scripts/training/run_lora_experiment_llmbased.sh
#
# The uniform-principle curriculum baseline is:
#   scripts/training/run_lora_experiment_uniform.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_lora_experiment_llmbased.sh" "$@"
