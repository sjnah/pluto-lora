#!/bin/bash
# Sequential quick-test suite launcher.
#
# Edit the RUN_* benchmark and method/version blocks below, then run:
#   bash scripts/evaluation/run_quick_test_suite.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIFIED_SCRIPT="${SCRIPT_DIR}/quick_test_unified.sh"

# Benchmark switches. Disabled by default to avoid accidental long evaluations.
export RUN_VAL14="${RUN_VAL14:-false}"
export RUN_VAL14_FAST="${RUN_VAL14_FAST:-false}"
export RUN_TEST14_HARD="${RUN_TEST14_HARD:-false}"
export RUN_TEST14_HARD_FAST="${RUN_TEST14_HARD_FAST:-true}"
export RUN_INTERPLAN10="${RUN_INTERPLAN10:-true}"
export RUN_INTERPLAN_BENCHMARK="${RUN_INTERPLAN_BENCHMARK:-false}"

# Method switches shared by all selected benchmarks.
export RUN_ZERO_SHOT="${RUN_ZERO_SHOT:-false}"
export RUN_RULE="${RUN_RULE:-true}"
export RUN_LOSS="${RUN_LOSS:-true}"
export RUN_UNIFORM="${RUN_UNIFORM:-true}"
export RUN_RANDOM_BUCKET="${RUN_RANDOM_BUCKET:-true}"
export RUN_LLM="${RUN_LLM:-true}"
export RUN_MPOC="${RUN_MPOC:-true}"

# Method versions shared by all selected benchmarks.
export LLM_VERSION="${LLM_VERSION:-v4.1.3.2.12}"
export RULE_VERSION="${RULE_VERSION:-v3.12}"
export LOSS_VERSION="${LOSS_VERSION:-v1.12}"
export RANDOM_BUCKET_VERSION="${RANDOM_BUCKET_VERSION:-v1.12}"
export MPOC_VERSION="${MPOC_VERSION:-v1.12}"
export UNIFORM_VERSION="${UNIFORM_VERSION:-v1.12}"

# Naming modes for outputs trained by the current suite.
export LLM_EXP_STYLE="${LLM_EXP_STYLE:-percentile_ehu}"
export UNIFORM_EXP_STYLE="${UNIFORM_EXP_STYLE:-uniform_only}"
export PERCENTILE_EHU_FINAL_PHASE="${PERCENTILE_EHU_FINAL_PHASE:-phaseC_uniform_consolidation}"

# Optional behavior.
export CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-false}"
export DRY_RUN="${DRY_RUN:-false}"

is_enabled() {
    case "$1" in
        true|TRUE|True|1|yes|YES|Yes|on|ON|On) return 0 ;;
        *) return 1 ;;
    esac
}

count_enabled_benchmarks() {
    local count=0
    is_enabled "$RUN_VAL14" && count=$((count + 1))
    is_enabled "$RUN_VAL14_FAST" && count=$((count + 1))
    is_enabled "$RUN_TEST14_HARD" && count=$((count + 1))
    is_enabled "$RUN_TEST14_HARD_FAST" && count=$((count + 1))
    is_enabled "$RUN_INTERPLAN10" && count=$((count + 1))
    is_enabled "$RUN_INTERPLAN_BENCHMARK" && count=$((count + 1))
    echo "$count"
}

run_one_benchmark() {
    local test_name="$1"

    echo ""
    echo "=============================================="
    echo "Quick-test benchmark: ${test_name}"
    echo "=============================================="

    set +e
    bash "$UNIFIED_SCRIPT" "$test_name"
    local status=$?
    set -e
    if [ "$status" -ne 0 ]; then
        echo "Error: quick-test benchmark ${test_name} failed with status ${status}."
        if is_enabled "$CONTINUE_ON_FAILURE"; then
            return 0
        fi
        exit "$status"
    fi
}

if [ "$(count_enabled_benchmarks)" -eq 0 ]; then
    echo "Error: no benchmarks are enabled. Set at least one RUN_* benchmark flag to true."
    exit 1
fi

echo "PLUTO quick-test suite"
echo "Enabled benchmarks:"
is_enabled "$RUN_VAL14" && echo "  val14"
is_enabled "$RUN_VAL14_FAST" && echo "  val14-fast"
is_enabled "$RUN_TEST14_HARD" && echo "  test14-hard"
is_enabled "$RUN_TEST14_HARD_FAST" && echo "  test14-hard-fast"
is_enabled "$RUN_INTERPLAN10" && echo "  interplan10"
is_enabled "$RUN_INTERPLAN_BENCHMARK" && echo "  interplan-benchmark"
echo "Enabled methods:"
is_enabled "$RUN_ZERO_SHOT" && echo "  Zero-shot"
is_enabled "$RUN_RULE" && echo "  Rule:          ${RULE_VERSION}"
is_enabled "$RUN_LOSS" && echo "  Loss:          ${LOSS_VERSION}"
is_enabled "$RUN_UNIFORM" && echo "  Uniform:       ${UNIFORM_VERSION} (${UNIFORM_EXP_STYLE})"
is_enabled "$RUN_RANDOM_BUCKET" && echo "  RandomBucket:  ${RANDOM_BUCKET_VERSION}"
is_enabled "$RUN_LLM" && echo "  LLM:           ${LLM_VERSION} (${LLM_EXP_STYLE})"
is_enabled "$RUN_MPOC" && echo "  MPOC:          ${MPOC_VERSION}"

is_enabled "$RUN_VAL14" && run_one_benchmark val14
is_enabled "$RUN_VAL14_FAST" && run_one_benchmark val14-fast
is_enabled "$RUN_TEST14_HARD" && run_one_benchmark test14-hard
is_enabled "$RUN_TEST14_HARD_FAST" && run_one_benchmark test14-hard-fast
is_enabled "$RUN_INTERPLAN10" && run_one_benchmark interplan10
is_enabled "$RUN_INTERPLAN_BENCHMARK" && run_one_benchmark interplan-benchmark

echo ""
echo "All enabled quick-test benchmarks finished."
