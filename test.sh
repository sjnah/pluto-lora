#!/bin/bash
# PLUTO evaluation entrypoint.
#
# Environment knobs:
#   SCENARIOS_PER_STAGE=2000
#   BATCH_SIZE=200

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec bash scripts/evaluation/quick_test.sh "$@"
