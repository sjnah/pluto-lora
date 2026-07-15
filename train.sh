#!/bin/bash
# PLUTO training entrypoint.
#
# Usage:
#   bash train.sh
#   bash train.sh --mode train_only --arms rule_exact loss_exact

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec bash scripts/run_lora_benchmark.sh "$@"
