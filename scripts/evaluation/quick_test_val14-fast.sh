#!/bin/bash
# Quick wrapper for the Val14 fast filter.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

apply_cli_overrides() {
    for arg in "$@"; do
        case "$arg" in
            RUN_ZERO_SHOT=*|RUN_RULE_BASED=*|RUN_LOSS_BASED=*|RUN_UNIFORM=*|RUN_RANDOM_BUCKET=*|RUN_LLM_CURRICULUM=*|RUN_MPOC=*|\
            SCENARIOS_PER_STAGE=*|BATCH_SIZE=*|SIMULATION_TYPE=*|SIMULATION_VERBOSE=*|ENABLE_PROGRESS_BAR=*|\
            SIMULATION_WORKER=*|SIMULATION_WORKER_THREADS=*|SIMULATION_WORKER_MAX_WORKERS=*|SIMULATION_NUM_GPUS=*|SIMULATION_NUM_CPUS=*|SIMULATION_RAY_LOG_TO_DRIVER=*|\
            PLUTO_EVAL_ALLOW_WANDB=*|WANDB_DISABLED=*|PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=*|\
            LLM_CURRICULUM_VERSION=*|LLM_CURRICULUM_SLUG=*|LLM_CURRICULUM_EXP=*|\
            UNIFORM_CURRICULUM_VERSION=*|UNIFORM_CURRICULUM_SLUG=*|UNIFORM_CURRICULUM_EXP=*|\
            RULE_CURRICULUM_VERSION=*|RULE_CURRICULUM_SLUG=*|RULE_CURRICULUM_EXP=*|\
            LOSS_CURRICULUM_VERSION=*|LOSS_CURRICULUM_SLUG=*|LOSS_CURRICULUM_EXP=*|\
            RANDOM_BUCKET_CURRICULUM_VERSION=*|RANDOM_BUCKET_CURRICULUM_SLUG=*|RANDOM_BUCKET_CURRICULUM_EXP=*|\
            MPOC_CURRICULUM_VERSION=*|MPOC_CURRICULUM_SLUG=*|MPOC_CURRICULUM_EXP=*|\
            PERCENTILE_EHU_FINAL_PHASE=*)
                export "$arg"
                ;;
            *)
                echo "Error: unsupported argument: $arg"
                echo "Use KEY=value before the command, or one of the supported KEY=value overrides after the script."
                exit 1
                ;;
        esac
    done
}

apply_cli_overrides "$@"

export FILTER_NAME="val14-fast"
export EXPERIMENT_SUFFIX="val14_fast"
export TEST_LABEL="Val14 Fast"
export SCENARIO_BUILDER="nuplan_v1_1_val"
export COLLECT_TEST="val14-fast"

export SCENARIOS_PER_STAGE="${SCENARIOS_PER_STAGE:-auto}"

export RUN_ZERO_SHOT="${RUN_ZERO_SHOT:-false}"
export RUN_RULE_BASED="${RUN_RULE_BASED:-true}"
export RUN_LOSS_BASED="${RUN_LOSS_BASED:-false}"
export RUN_UNIFORM="${RUN_UNIFORM:-true}"
export RUN_RANDOM_BUCKET="${RUN_RANDOM_BUCKET:-false}"
export RUN_LLM_CURRICULUM="${RUN_LLM_CURRICULUM:-true}"
export RUN_MPOC="${RUN_MPOC:-true}"

exec "${SCRIPT_DIR}/quick_test_val14.sh"
