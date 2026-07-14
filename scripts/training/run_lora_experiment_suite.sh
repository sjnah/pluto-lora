#!/bin/bash
# Sequential PLUTO LoRA experiment launcher. Defaults are resolved from
# EXPERIMENT_SUITE_CONFIG; environment variables are one-run overrides.
#
# Run:
#   bash scripts/training/run_lora_experiment_suite.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

LORA_EXPERIMENT_SCRIPT="${SCRIPT_DIR}/run_lora_experiment.sh"
CONFIG_RESOLVER="${SCRIPT_DIR}/resolve_lora_experiment_config.py"
EXPERIMENT_SUITE_CONFIG="${EXPERIMENT_SUITE_CONFIG:-${REPO_ROOT}/config/experiment_suite/flat_lr_comparison_v1.yaml}"

eval "$(python3 "$CONFIG_RESOLVER" --suite "$EXPERIMENT_SUITE_CONFIG" --format shell)"
TRAINING_PROTOCOL_CONFIG="${TRAINING_PROTOCOL_CONFIG:-$CFG_SUITE_TRAINING_PROTOCOL}"
eval "$(python3 "$CONFIG_RESOLVER" \
    --protocol "$TRAINING_PROTOCOL_CONFIG" \
    --method "${REPO_ROOT}/config/curriculum_method/llm.yaml" \
    --format shell)"
TRAINING_PROTOCOL_ID="$CFG_PROTOCOL_ID"

# Method switches from the selected suite.
export RUN_LLM="${RUN_LLM:-$CFG_SUITE_RUN_LLM}"
export RUN_RULE="${RUN_RULE:-$CFG_SUITE_RUN_RULE}"
export RUN_LOSS="${RUN_LOSS:-$CFG_SUITE_RUN_LOSS}"
export RUN_RANDOM_BUCKET="${RUN_RANDOM_BUCKET:-$CFG_SUITE_RUN_RANDOM}"
export RUN_MPOC="${RUN_MPOC:-$CFG_SUITE_RUN_MPOC}"
export RUN_UNIFORM="${RUN_UNIFORM:-$CFG_SUITE_RUN_UNIFORM}"

# Method versions. These feed CURRICULUM_VERSION for percentile-EHU methods.
export LLM_VERSION="${LLM_VERSION:-$CFG_SUITE_LLM_VERSION}"
export RULE_VERSION="${RULE_VERSION:-$CFG_SUITE_RULE_VERSION}"
export LOSS_VERSION="${LOSS_VERSION:-$CFG_SUITE_LOSS_VERSION}"
export RANDOM_BUCKET_VERSION="${RANDOM_BUCKET_VERSION:-$CFG_SUITE_RANDOM_VERSION}"
export MPOC_VERSION="${MPOC_VERSION:-$CFG_SUITE_MPOC_VERSION}"
export UNIFORM_VERSION="${UNIFORM_VERSION:-$CFG_SUITE_UNIFORM_VERSION}"

# Optional controls.
export CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-$CFG_SUITE_CONTINUE_ON_FAILURE}"
export DRY_RUN="${DRY_RUN:-false}"
export TYPE_ROUTING_MODE="${TYPE_ROUTING_MODE:-$CFG_SUITE_TYPE_ROUTING_MODE}"
export FEATURE_CACHE_NAME="${FEATURE_CACHE_NAME:-$CFG_SUITE_FEATURE_CACHE_NAME}"
export RUN_LLM_TYPE_ROUTING_COMPARISON="${RUN_LLM_TYPE_ROUTING_COMPARISON:-false}"
export LLM_TYPE_ROUTING_OFF_BASE_EXP="${LLM_TYPE_ROUTING_OFF_BASE_EXP:-curriculum_lora_llm_percentile_ehu_${LLM_VERSION}_${TRAINING_PROTOCOL_ID}_type_off}"
export LLM_TYPE_ROUTING_ON_BASE_EXP="${LLM_TYPE_ROUTING_ON_BASE_EXP:-curriculum_lora_llm_percentile_ehu_${LLM_VERSION}_${TRAINING_PROTOCOL_ID}_type_on}"

is_enabled() {
    case "$1" in
        true|TRUE|True|1|yes|YES|Yes|on|ON|On) return 0 ;;
        *) return 1 ;;
    esac
}

count_enabled_methods() {
    local count=0
    is_enabled "$RUN_LLM" && count=$((count + 1))
    is_enabled "$RUN_RULE" && count=$((count + 1))
    is_enabled "$RUN_LOSS" && count=$((count + 1))
    is_enabled "$RUN_RANDOM_BUCKET" && count=$((count + 1))
    is_enabled "$RUN_MPOC" && count=$((count + 1))
    is_enabled "$RUN_UNIFORM" && count=$((count + 1))
    echo "$count"
}

run_percentile_ehu_method() {
    local flag_value="$1"
    local method="$2"
    local label="$3"
    local version="$4"
    local base_exp_var="$5"
    local filter_prefix_var="$6"
    local type_routing_mode="${7:-off}"
    local sampler_mode_override="${8:-}"

    if ! is_enabled "$flag_value"; then
        return 0
    fi

    local base_exp="${!base_exp_var:-}"
    local filter_prefix="${!filter_prefix_var:-}"
    local command_label="${label} percentile-EHU (${version})"

    if is_enabled "$DRY_RUN"; then
        echo ""
        echo "=============================================="
        echo "$command_label"
        echo "=============================================="
        echo "DRY_RUN: METHOD=${method} METHOD_LABEL=${label} CURRICULUM_VERSION=${version} TRAINING_PROTOCOL_CONFIG=${TRAINING_PROTOCOL_CONFIG} bash ${LORA_EXPERIMENT_SCRIPT}"
        echo "         TYPE_ROUTING_MODE=${type_routing_mode}${sampler_mode_override:+ SAMPLER_MODE=${sampler_mode_override}}"
        echo "         FEATURE_CACHE_NAME=${FEATURE_CACHE_NAME}"
        [ -n "$base_exp" ] && echo "         CURRICULUM_BASE_EXP=${base_exp}"
        [ -n "$filter_prefix" ] && echo "         FILTER_PREFIX=${filter_prefix}"
        return 0
    fi

    echo ""
    echo "=============================================="
    echo "$command_label"
    echo "=============================================="
    set +e
    (
        set -e
        unset CURRICULUM_BASE_EXP FILTER_PREFIX
        [ -n "$base_exp" ] && export CURRICULUM_BASE_EXP="$base_exp"
        [ -n "$filter_prefix" ] && export FILTER_PREFIX="$filter_prefix"
        export CURRICULUM_MODE=percentile_ehu
        export METHOD="$method"
        export METHOD_LABEL="$label"
        export CURRICULUM_VERSION="$version"
        export TRAINING_PROTOCOL_CONFIG
        export TYPE_ROUTING_MODE="$type_routing_mode"
        [ -n "$sampler_mode_override" ] && export SAMPLER_MODE="$sampler_mode_override"
        bash "$LORA_EXPERIMENT_SCRIPT"
    )
    local status=$?
    set -e
    if [ "$status" -ne 0 ]; then
        echo "Error: ${command_label} failed with status ${status}."
        if is_enabled "$CONTINUE_ON_FAILURE"; then
            return 0
        fi
        exit "$status"
    fi
}

run_uniform_method() {
    if ! is_enabled "$RUN_UNIFORM"; then
        return 0
    fi

    if is_enabled "$DRY_RUN"; then
        echo ""
        echo "=============================================="
        echo "Uniform FT (${UNIFORM_VERSION})"
        echo "=============================================="
        echo "DRY_RUN: METHOD=uniform CURRICULUM_VERSION=${UNIFORM_VERSION} TRAINING_PROTOCOL_CONFIG=${TRAINING_PROTOCOL_CONFIG} bash ${LORA_EXPERIMENT_SCRIPT}"
        echo "         FEATURE_CACHE_NAME=${FEATURE_CACHE_NAME}"
        [ -n "${UNIFORM_CURRICULUM_BASE_EXP:-}" ] && echo "         CURRICULUM_BASE_EXP=${UNIFORM_CURRICULUM_BASE_EXP}"
        return 0
    fi

    echo ""
    echo "=============================================="
    echo "Uniform FT (${UNIFORM_VERSION})"
    echo "=============================================="
    set +e
    (
        set -e
        unset CURRICULUM_BASE_EXP
        [ -n "${UNIFORM_CURRICULUM_BASE_EXP:-}" ] && export CURRICULUM_BASE_EXP="$UNIFORM_CURRICULUM_BASE_EXP"
        export METHOD=uniform
        export CURRICULUM_VERSION="$UNIFORM_VERSION"
        export TRAINING_PROTOCOL_CONFIG
        bash "$LORA_EXPERIMENT_SCRIPT"
    )
    local status=$?
    set -e
    if [ "$status" -ne 0 ]; then
        echo "Error: Uniform FT (${UNIFORM_VERSION}) failed with status ${status}."
        if is_enabled "$CONTINUE_ON_FAILURE"; then
            return 0
        fi
        exit "$status"
    fi
}

if [ "$(count_enabled_methods)" -eq 0 ]; then
    echo "Error: no training methods are enabled. Set at least one RUN_* variable to true."
    exit 1
fi

echo "PLUTO LoRA experiment suite"
echo "Suite: ${CFG_SUITE_ID} (${CFG_SUITE_PATH})"
echo "Training protocol: ${TRAINING_PROTOCOL_ID} (${TRAINING_PROTOCOL_CONFIG})"
echo "Feature cache: ${FEATURE_CACHE_NAME}"
echo "Enabled methods:"
is_enabled "$RUN_LLM" && echo "  LLM:           ${LLM_VERSION}"
is_enabled "$RUN_RULE" && echo "  Rule:          ${RULE_VERSION}"
is_enabled "$RUN_LOSS" && echo "  Loss:          ${LOSS_VERSION}"
is_enabled "$RUN_RANDOM_BUCKET" && echo "  RandomBucket:  ${RANDOM_BUCKET_VERSION}"
is_enabled "$RUN_MPOC" && echo "  MPOC:          ${MPOC_VERSION}"
is_enabled "$RUN_UNIFORM" && echo "  Uniform:       ${UNIFORM_VERSION}"

if is_enabled "$RUN_LLM" && is_enabled "$RUN_LLM_TYPE_ROUTING_COMPARISON"; then
    run_percentile_ehu_method "$RUN_LLM" llm "LLM-guided type-off" "$LLM_VERSION" LLM_TYPE_ROUTING_OFF_BASE_EXP LLM_FILTER_PREFIX off legacy_weighted
    run_percentile_ehu_method "$RUN_LLM" llm "LLM-guided type-on" "$LLM_VERSION" LLM_TYPE_ROUTING_ON_BASE_EXP LLM_FILTER_PREFIX on legacy_weighted
else
    case "${TYPE_ROUTING_MODE:-off}" in
        on|enabled)
            run_percentile_ehu_method "$RUN_LLM" llm "LLM-guided type-on" "$LLM_VERSION" LLM_TYPE_ROUTING_ON_BASE_EXP LLM_FILTER_PREFIX on legacy_weighted
            ;;
        *)
            run_percentile_ehu_method "$RUN_LLM" llm "LLM-guided" "$LLM_VERSION" LLM_CURRICULUM_BASE_EXP LLM_FILTER_PREFIX off
            ;;
    esac
fi
run_percentile_ehu_method "$RUN_RULE" rule "Rule-based" "$RULE_VERSION" RULE_CURRICULUM_BASE_EXP RULE_FILTER_PREFIX
run_percentile_ehu_method "$RUN_LOSS" loss "Loss-ranked" "$LOSS_VERSION" LOSS_CURRICULUM_BASE_EXP LOSS_FILTER_PREFIX
run_percentile_ehu_method "$RUN_RANDOM_BUCKET" random "RandomBucket" "$RANDOM_BUCKET_VERSION" RANDOM_BUCKET_CURRICULUM_BASE_EXP RANDOM_BUCKET_FILTER_PREFIX
run_percentile_ehu_method "$RUN_MPOC" mpoc "MPOC" "$MPOC_VERSION" MPOC_CURRICULUM_BASE_EXP MPOC_FILTER_PREFIX
run_uniform_method

echo ""
echo "All enabled LoRA experiments finished."
