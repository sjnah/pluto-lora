#!/bin/bash
# Quick wrapper for the Val14 fast filter.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export FILTER_NAME="val14-fast"
export EXPERIMENT_SUFFIX="val14_fast"
export TEST_LABEL="Val14 Fast"
export SCENARIO_BUILDER="nuplan_v1_1_val"
export SCENARIOS_PER_STAGE="${SCENARIOS_PER_STAGE:-270}"

export RUN_ZERO_SHOT="${RUN_ZERO_SHOT:-false}"
export RUN_RULE_BASED="${RUN_RULE_BASED:-false}"
export RUN_LOSS_BASED="${RUN_LOSS_BASED:-false}"
export RUN_UNIFORM="${RUN_UNIFORM:-false}"
export RUN_RANDOM_BUCKET="${RUN_RANDOM_BUCKET:-false}"
export RUN_LLM_CURRICULUM="${RUN_LLM_CURRICULUM:-false}"
export RUN_MPOC="${RUN_MPOC:-true}"

exec "${SCRIPT_DIR}/quick_test_val14.sh"
