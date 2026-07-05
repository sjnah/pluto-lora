#!/bin/bash
################################################################################
# Quick Test: Test14-Hard Fast Filter
# Uses test14-hard-fast.yaml scenario filter with nuplan-v1.1_test dataset.
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export FILTER_NAME="test14-hard-fast"
export EXPERIMENT_SUFFIX="test14_hard_fast"
export TEST_LABEL="Test14-Hard Fast"
export SCENARIO_BUILDER="nuplan_v1_1_test"
export COLLECT_TEST="test14-hard-fast"

export SCENARIOS_PER_STAGE="${SCENARIOS_PER_STAGE:-auto}"
export RUN_ZERO_SHOT="${RUN_ZERO_SHOT:-false}"
export RUN_RULE_BASED="${RUN_RULE_BASED:-false}"
export RUN_LOSS_BASED="${RUN_LOSS_BASED:-false}"
export RUN_UNIFORM="${RUN_UNIFORM:-false}"
export RUN_RANDOM_BUCKET="${RUN_RANDOM_BUCKET:-false}"
export RUN_LLM_CURRICULUM="${RUN_LLM_CURRICULUM:-false}"
export RUN_MPOC="${RUN_MPOC:-true}"

exec "${SCRIPT_DIR}/quick_test_test14-hard.sh"
