#!/bin/bash
# PLUTO paper-benchmark evaluation-only entrypoint.
# Arm, seed, benchmark, and checkpoint selection come from the benchmark YAML;
# optional CLI flags can override them for one invocation.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec bash scripts/run_lora_benchmark.sh --mode evaluate_only "$@"
