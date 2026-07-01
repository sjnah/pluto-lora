#!/bin/bash
# PLUTO training entrypoint.
#
# Usage:
#   bash train.sh            # LLM-percentile LoRA experiment
#   bash train.sh llm        # same as default
#   bash train.sh rulebased  # rule-based score comparison

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-llm}"

case "$MODE" in
    llm|percentile)
        exec bash scripts/training/run_lora_experiment.sh
        ;;
    rulebased|rule)
        exec bash scripts/training/run_lora_experiment_rulebased.sh
        ;;
    *)
        echo "Unknown training mode: $MODE"
        echo "Usage: bash train.sh [llm|rulebased]"
        exit 2
        ;;
esac
