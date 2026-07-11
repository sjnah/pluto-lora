#!/bin/bash
# Sequential PLUTO LoRA experiment launcher.
#
# Edit the RUN_* and *_VERSION block below, then run:
#   bash scripts/training/run_lora_experiment_suite.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

PERCENTILE_EHU_SCRIPT="${SCRIPT_DIR}/run_lora_experiment_percentile_ehu.sh"
UNIFORM_SCRIPT="${SCRIPT_DIR}/run_lora_experiment_uniform.sh"

# Method switches. Disabled by default to avoid accidental long training runs.
export RUN_LLM="${RUN_LLM:-true}"
export RUN_RULE="${RUN_RULE:-true}"
export RUN_LOSS="${RUN_LOSS:-true}"
export RUN_RANDOM_BUCKET="${RUN_RANDOM_BUCKET:-true}"
export RUN_MPOC="${RUN_MPOC:-true}"
export RUN_UNIFORM="${RUN_UNIFORM:-true}"

# Method versions. These feed CURRICULUM_VERSION for percentile-EHU methods.
export LLM_VERSION="${LLM_VERSION:-v4.1.3.2.12}"
export RULE_VERSION="${RULE_VERSION:-v3.12}"
export LOSS_VERSION="${LOSS_VERSION:-v1.12}"
export RANDOM_BUCKET_VERSION="${RANDOM_BUCKET_VERSION:-v1.12}"
export MPOC_VERSION="${MPOC_VERSION:-v1.12}"
export UNIFORM_VERSION="${UNIFORM_VERSION:-v1.12}"

# Optional controls.
export CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-false}"
export DRY_RUN="${DRY_RUN:-false}"

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
        echo "DRY_RUN: METHOD=${method} METHOD_LABEL=${label} CURRICULUM_VERSION=${version} bash ${PERCENTILE_EHU_SCRIPT}"
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
        bash "$PERCENTILE_EHU_SCRIPT"
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
        echo "DRY_RUN: UNIFORM_CURRICULUM_VERSION=${UNIFORM_VERSION} bash ${UNIFORM_SCRIPT}"
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
        export UNIFORM_CURRICULUM_VERSION="$UNIFORM_VERSION"
        bash "$UNIFORM_SCRIPT"
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
echo "Enabled methods:"
is_enabled "$RUN_LLM" && echo "  LLM:           ${LLM_VERSION}"
is_enabled "$RUN_RULE" && echo "  Rule:          ${RULE_VERSION}"
is_enabled "$RUN_LOSS" && echo "  Loss:          ${LOSS_VERSION}"
is_enabled "$RUN_RANDOM_BUCKET" && echo "  RandomBucket:  ${RANDOM_BUCKET_VERSION}"
is_enabled "$RUN_MPOC" && echo "  MPOC:          ${MPOC_VERSION}"
is_enabled "$RUN_UNIFORM" && echo "  Uniform:       ${UNIFORM_VERSION}"

run_percentile_ehu_method "$RUN_LLM" llm "LLM-guided" "$LLM_VERSION" LLM_CURRICULUM_BASE_EXP LLM_FILTER_PREFIX
run_percentile_ehu_method "$RUN_RULE" rule "Rule-based" "$RULE_VERSION" RULE_CURRICULUM_BASE_EXP RULE_FILTER_PREFIX
run_percentile_ehu_method "$RUN_LOSS" loss "Loss-ranked" "$LOSS_VERSION" LOSS_CURRICULUM_BASE_EXP LOSS_FILTER_PREFIX
run_percentile_ehu_method "$RUN_RANDOM_BUCKET" random "RandomBucket" "$RANDOM_BUCKET_VERSION" RANDOM_BUCKET_CURRICULUM_BASE_EXP RANDOM_BUCKET_FILTER_PREFIX
run_percentile_ehu_method "$RUN_MPOC" mpoc "MPOC" "$MPOC_VERSION" MPOC_CURRICULUM_BASE_EXP MPOC_FILTER_PREFIX
run_uniform_method

echo ""
echo "All enabled LoRA experiments finished."
