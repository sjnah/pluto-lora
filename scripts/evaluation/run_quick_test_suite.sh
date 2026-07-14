#!/bin/bash
# Sequential quick-test suite launcher. Defaults are resolved from
# EXPERIMENT_SUITE_CONFIG; environment variables are one-run overrides.
#
# Run:
#   bash scripts/evaluation/run_quick_test_suite.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UNIFIED_SCRIPT="${SCRIPT_DIR}/quick_test_unified.sh"
CONFIG_RESOLVER="${REPO_ROOT}/scripts/training/resolve_lora_experiment_config.py"

EXPERIMENT_SUITE_CONFIG="${EXPERIMENT_SUITE_CONFIG:-${REPO_ROOT}/config/experiment_suite/flat_lr_comparison_v1.yaml}"
eval "$(python3 "$CONFIG_RESOLVER" --suite "$EXPERIMENT_SUITE_CONFIG" --format shell)"
TRAINING_PROTOCOL_CONFIG="${TRAINING_PROTOCOL_CONFIG:-$CFG_SUITE_TRAINING_PROTOCOL}"
eval "$(python3 "$CONFIG_RESOLVER" \
    --protocol "$TRAINING_PROTOCOL_CONFIG" \
    --method "${REPO_ROOT}/config/curriculum_method/llm.yaml" \
    --format shell)"
export TRAINING_PROTOCOL_ID="${TRAINING_PROTOCOL_ID:-$CFG_PROTOCOL_ID}"

# Benchmark switches from the selected suite.
export RUN_VAL14="${RUN_VAL14:-$CFG_SUITE_RUN_VAL14}"
export RUN_VAL14_FAST="${RUN_VAL14_FAST:-$CFG_SUITE_RUN_VAL14_FAST}"
export RUN_TEST14_HARD="${RUN_TEST14_HARD:-$CFG_SUITE_RUN_TEST14_HARD}"
export RUN_TEST14_HARD_FAST="${RUN_TEST14_HARD_FAST:-$CFG_SUITE_RUN_TEST14_HARD_FAST}"
export RUN_INTERPLAN10="${RUN_INTERPLAN10:-$CFG_SUITE_RUN_INTERPLAN10}"
export RUN_INTERPLAN_BENCHMARK="${RUN_INTERPLAN_BENCHMARK:-$CFG_SUITE_RUN_INTERPLAN_BENCHMARK}"

# Method switches shared by all selected benchmarks.
export RUN_ZERO_SHOT="${RUN_ZERO_SHOT:-false}"
export RUN_RULE="${RUN_RULE:-$CFG_SUITE_RUN_RULE}"
export RUN_LOSS="${RUN_LOSS:-$CFG_SUITE_RUN_LOSS}"
export RUN_UNIFORM="${RUN_UNIFORM:-$CFG_SUITE_RUN_UNIFORM}"
export RUN_RANDOM_BUCKET="${RUN_RANDOM_BUCKET:-$CFG_SUITE_RUN_RANDOM}"
export RUN_LLM="${RUN_LLM:-$CFG_SUITE_RUN_LLM}"
export RUN_MPOC="${RUN_MPOC:-$CFG_SUITE_RUN_MPOC}"

# Method versions shared by all selected benchmarks.
export LLM_VERSION="${LLM_VERSION:-$CFG_SUITE_LLM_VERSION}"
export RULE_VERSION="${RULE_VERSION:-$CFG_SUITE_RULE_VERSION}"
export LOSS_VERSION="${LOSS_VERSION:-$CFG_SUITE_LOSS_VERSION}"
export RANDOM_BUCKET_VERSION="${RANDOM_BUCKET_VERSION:-$CFG_SUITE_RANDOM_VERSION}"
export MPOC_VERSION="${MPOC_VERSION:-$CFG_SUITE_MPOC_VERSION}"
export UNIFORM_VERSION="${UNIFORM_VERSION:-$CFG_SUITE_UNIFORM_VERSION}"

# Naming modes for outputs trained by the current suite.
export LLM_EXP_STYLE="${LLM_EXP_STYLE:-percentile_ehu}"
export UNIFORM_EXP_STYLE="${UNIFORM_EXP_STYLE:-uniform_only}"
export PERCENTILE_EHU_FINAL_PHASE="${PERCENTILE_EHU_FINAL_PHASE:-phaseC_uniform_consolidation}"

# Optional behavior.
export CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-$CFG_SUITE_CONTINUE_ON_FAILURE}"
export DRY_RUN="${DRY_RUN:-false}"
export RUN_LLM_TYPE_ROUTING_COMPARISON="${RUN_LLM_TYPE_ROUTING_COMPARISON:-false}"
export TYPE_ROUTING_MODE="${TYPE_ROUTING_MODE:-$CFG_SUITE_TYPE_ROUTING_MODE}"

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

run_benchmark_variant() {
    local test_name="$1"
    local variant_label="$2"
    local llm_slug="${3:-}"
    local llm_exp="${4:-}"
    local llm_only="${5:-false}"

    echo ""
    echo "=============================================="
    echo "Quick-test benchmark: ${test_name}${variant_label:+ (${variant_label})}"
    echo "=============================================="

    set +e
    if is_enabled "$llm_only"; then
        env \
            RUN_ZERO_SHOT=false RUN_RULE=false RUN_LOSS=false RUN_UNIFORM=false \
            RUN_RANDOM_BUCKET=false RUN_LLM=true RUN_MPOC=false \
            LLM_CURRICULUM_SLUG="$llm_slug" \
            LLM_CURRICULUM_EXP="$llm_exp" \
            bash "$UNIFIED_SCRIPT" "$test_name"
    elif [ -n "$llm_slug" ]; then
        env \
            LLM_CURRICULUM_SLUG="$llm_slug" \
            LLM_CURRICULUM_EXP="$llm_exp" \
            bash "$UNIFIED_SCRIPT" "$test_name"
    else
        bash "$UNIFIED_SCRIPT" "$test_name"
    fi
    local status=$?
    set -e
    if [ "$status" -ne 0 ]; then
        echo "Error: quick-test benchmark ${test_name}${variant_label:+ (${variant_label})} failed with status ${status}."
        if is_enabled "$CONTINUE_ON_FAILURE"; then
            return 0
        fi
        exit "$status"
    fi
}

run_one_benchmark() {
    local test_name="$1"
    if is_enabled "$RUN_LLM" && is_enabled "$RUN_LLM_TYPE_ROUTING_COMPARISON"; then
        local off_slug="curriculum_llm_percentile_ehu_${LLM_VERSION}_${TRAINING_PROTOCOL_ID}_type_off"
        local on_slug="curriculum_llm_percentile_ehu_${LLM_VERSION}_${TRAINING_PROTOCOL_ID}_type_on"
        local off_exp="curriculum_lora_llm_percentile_ehu_${LLM_VERSION}_${TRAINING_PROTOCOL_ID}_type_off_${PERCENTILE_EHU_FINAL_PHASE}"
        local on_exp="curriculum_lora_llm_percentile_ehu_${LLM_VERSION}_${TRAINING_PROTOCOL_ID}_type_on_${PERCENTILE_EHU_FINAL_PHASE}"
        # Run all requested comparison methods with type-off, then only the
        # second LLM variant so non-LLM baselines are not duplicated.
        run_benchmark_variant "$test_name" "type-off" "$off_slug" "$off_exp" false
        run_benchmark_variant "$test_name" "type-on" "$on_slug" "$on_exp" true
    elif is_enabled "$RUN_LLM" && { [ "${TYPE_ROUTING_MODE:-off}" = "on" ] || [ "${TYPE_ROUTING_MODE:-off}" = "enabled" ]; }; then
        local on_slug="${LLM_CURRICULUM_SLUG:-curriculum_llm_percentile_ehu_${LLM_VERSION}_${TRAINING_PROTOCOL_ID}_type_on}"
        local on_exp="${LLM_CURRICULUM_EXP:-curriculum_lora_llm_percentile_ehu_${LLM_VERSION}_${TRAINING_PROTOCOL_ID}_type_on_${PERCENTILE_EHU_FINAL_PHASE}}"
        run_benchmark_variant "$test_name" "type-on" "$on_slug" "$on_exp" false
    else
        run_benchmark_variant "$test_name" "" "" "" false
    fi
}

if [ "$(count_enabled_benchmarks)" -eq 0 ]; then
    echo "Error: no benchmarks are enabled. Set at least one RUN_* benchmark flag to true."
    exit 1
fi

echo "PLUTO quick-test suite"
echo "Training protocol: ${TRAINING_PROTOCOL_ID}"
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
is_enabled "$RUN_LLM" && is_enabled "$RUN_LLM_TYPE_ROUTING_COMPARISON" && echo "    variants:    type-off, type-on (separate slugs/results)"
is_enabled "$RUN_MPOC" && echo "  MPOC:          ${MPOC_VERSION}"

if is_enabled "$RUN_LLM" && { is_enabled "$RUN_LLM_TYPE_ROUTING_COMPARISON" || [ "${TYPE_ROUTING_MODE:-off}" = "on" ] || [ "${TYPE_ROUTING_MODE:-off}" = "enabled" ]; } && [ "$LLM_EXP_STYLE" != "percentile_ehu" ]; then
    echo "Error: type-routing quick tests require LLM_EXP_STYLE=percentile_ehu" >&2
    exit 1
fi

is_enabled "$RUN_VAL14" && run_one_benchmark val14
is_enabled "$RUN_VAL14_FAST" && run_one_benchmark val14-fast
is_enabled "$RUN_TEST14_HARD" && run_one_benchmark test14-hard
is_enabled "$RUN_TEST14_HARD_FAST" && run_one_benchmark test14-hard-fast
is_enabled "$RUN_INTERPLAN10" && run_one_benchmark interplan10
is_enabled "$RUN_INTERPLAN_BENCHMARK" && run_one_benchmark interplan-benchmark

echo ""
echo "All enabled quick-test benchmarks finished."
