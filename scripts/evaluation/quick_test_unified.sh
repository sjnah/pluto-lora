#!/bin/bash
# Dispatch one PLUTO quick-test benchmark with shared method/version controls.
#
# Example:
#   bash scripts/evaluation/quick_test_unified.sh test14-hard-fast RUN_LLM=true LLM_VERSION=v4.3.13

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

POSITIONAL_TEST=""

for arg in "$@"; do
    case "$arg" in
        *=*)
            export "$arg"
            ;;
        *)
            if [ -n "$POSITIONAL_TEST" ]; then
                echo "Error: multiple tests provided: ${POSITIONAL_TEST} and ${arg}"
                exit 1
            fi
            POSITIONAL_TEST="$arg"
            ;;
    esac
done

TEST_NAME="${TEST_NAME:-${TEST:-${POSITIONAL_TEST:-}}}"
if [ -z "$TEST_NAME" ]; then
    echo "Error: set TEST_NAME or pass one test name."
    echo "Tests: val14, val14-fast, test14-hard, test14-hard-fast, interplan10"
    exit 1
fi

is_enabled() {
    case "$1" in
        true|TRUE|True|1|yes|YES|Yes|on|ON|On) return 0 ;;
        *) return 1 ;;
    esac
}

# Canonical method switches for this unified wrapper.
export RUN_ZERO_SHOT="${RUN_ZERO_SHOT:-false}"
export RUN_RULE="${RUN_RULE:-${RUN_RULE_BASED:-false}}"
export RUN_LOSS="${RUN_LOSS:-${RUN_LOSS_BASED:-false}}"
export RUN_UNIFORM="${RUN_UNIFORM:-false}"
export RUN_RANDOM_BUCKET="${RUN_RANDOM_BUCKET:-false}"
export RUN_LLM="${RUN_LLM:-${RUN_LLM_CURRICULUM:-false}}"
export RUN_MPOC="${RUN_MPOC:-false}"

# Map unified switches to the underlying quick-test entrypoint contract.
export RUN_RULE_BASED="$RUN_RULE"
export RUN_LOSS_BASED="$RUN_LOSS"
export RUN_LLM_CURRICULUM="$RUN_LLM"

# Method versions. The suite defaults to the current percentile-EHU generation.
export PERCENTILE_EHU_FINAL_PHASE="${PERCENTILE_EHU_FINAL_PHASE:-phaseC_uniform_consolidation}"
export LLM_VERSION="${LLM_VERSION:-${LLM_CURRICULUM_VERSION:-v4.3.13}}"
export RULE_VERSION="${RULE_VERSION:-${RULE_CURRICULUM_VERSION:-v4.3.13}}"
export LOSS_VERSION="${LOSS_VERSION:-${LOSS_CURRICULUM_VERSION:-v4.3.13}}"
export RANDOM_BUCKET_VERSION="${RANDOM_BUCKET_VERSION:-${RANDOM_BUCKET_CURRICULUM_VERSION:-v4.3.13}}"
export MPOC_VERSION="${MPOC_VERSION:-${MPOC_CURRICULUM_VERSION:-v4.3.13}}"
export UNIFORM_VERSION="${UNIFORM_VERSION:-${UNIFORM_CURRICULUM_VERSION:-v4.3.13}}"

export RULE_CURRICULUM_VERSION="${RULE_CURRICULUM_VERSION:-$RULE_VERSION}"
export LOSS_CURRICULUM_VERSION="${LOSS_CURRICULUM_VERSION:-$LOSS_VERSION}"
export RANDOM_BUCKET_CURRICULUM_VERSION="${RANDOM_BUCKET_CURRICULUM_VERSION:-$RANDOM_BUCKET_VERSION}"
export MPOC_CURRICULUM_VERSION="${MPOC_CURRICULUM_VERSION:-$MPOC_VERSION}"
export RULE_CURRICULUM_SLUG="${RULE_CURRICULUM_SLUG:-curriculum_rule_percentile_ehu_${RULE_VERSION}}"
export RULE_CURRICULUM_EXP="${RULE_CURRICULUM_EXP:-curriculum_lora_rule_percentile_ehu_${RULE_VERSION}_${PERCENTILE_EHU_FINAL_PHASE}}"
export LOSS_CURRICULUM_SLUG="${LOSS_CURRICULUM_SLUG:-curriculum_loss_percentile_ehu_${LOSS_VERSION}}"
export LOSS_CURRICULUM_EXP="${LOSS_CURRICULUM_EXP:-curriculum_lora_loss_percentile_ehu_${LOSS_VERSION}_${PERCENTILE_EHU_FINAL_PHASE}}"
export RANDOM_BUCKET_CURRICULUM_SLUG="${RANDOM_BUCKET_CURRICULUM_SLUG:-curriculum_randombucket_percentile_ehu_${RANDOM_BUCKET_VERSION}}"
export RANDOM_BUCKET_CURRICULUM_EXP="${RANDOM_BUCKET_CURRICULUM_EXP:-curriculum_lora_random_percentile_ehu_${RANDOM_BUCKET_VERSION}_${PERCENTILE_EHU_FINAL_PHASE}}"
export MPOC_CURRICULUM_SLUG="${MPOC_CURRICULUM_SLUG:-curriculum_mpoc_percentile_ehu_${MPOC_VERSION}}"
export MPOC_CURRICULUM_EXP="${MPOC_CURRICULUM_EXP:-curriculum_lora_mpoc_percentile_ehu_${MPOC_VERSION}_${PERCENTILE_EHU_FINAL_PHASE}}"

# LLM has legacy and percentile-EHU naming in circulation. The unified wrapper
# defaults to the percentile-EHU output produced by run_lora_experiment_suite.sh.
export LLM_EXP_STYLE="${LLM_EXP_STYLE:-percentile_ehu}" # percentile_ehu or legacy_guided
case "$LLM_EXP_STYLE" in
    percentile_ehu)
        export LLM_CURRICULUM_VERSION="${LLM_CURRICULUM_VERSION:-$LLM_VERSION}"
        export LLM_CURRICULUM_SLUG="${LLM_CURRICULUM_SLUG:-curriculum_llm_percentile_ehu_${LLM_VERSION}}"
        export LLM_CURRICULUM_EXP="${LLM_CURRICULUM_EXP:-curriculum_lora_llm_percentile_ehu_${LLM_VERSION}_${PERCENTILE_EHU_FINAL_PHASE}}"
        ;;
    legacy_guided)
        export LLM_CURRICULUM_VERSION="${LLM_CURRICULUM_VERSION:-$LLM_VERSION}"
        export LLM_CURRICULUM_SLUG="${LLM_CURRICULUM_SLUG:-curriculum_llm_guided_${LLM_VERSION}}"
        export LLM_CURRICULUM_EXP="${LLM_CURRICULUM_EXP:-curriculum_lora_llm_guided_${LLM_VERSION}_stage3_high}"
        ;;
    *)
        echo "Error: unsupported LLM_EXP_STYLE=${LLM_EXP_STYLE}. Use percentile_ehu or legacy_guided."
        exit 1
        ;;
esac

# The current uniform training script uses the *_uniform_only_* experiment base.
# Set UNIFORM_EXP_STYLE=legacy_uniform for older curriculum_lora_uniform_* runs.
export UNIFORM_EXP_STYLE="${UNIFORM_EXP_STYLE:-uniform_only}" # uniform_only or legacy_uniform
export UNIFORM_CURRICULUM_VERSION="${UNIFORM_CURRICULUM_VERSION:-$UNIFORM_VERSION}"
export UNIFORM_CURRICULUM_SLUG="${UNIFORM_CURRICULUM_SLUG:-curriculum_uniform_${UNIFORM_VERSION}}"
export DRY_RUN="${DRY_RUN:-false}"
case "$UNIFORM_EXP_STYLE" in
    uniform_only)
        export UNIFORM_CURRICULUM_EXP="${UNIFORM_CURRICULUM_EXP:-curriculum_lora_uniform_only_${UNIFORM_VERSION}_stage3_uniform}"
        ;;
    legacy_uniform)
        export UNIFORM_CURRICULUM_EXP="${UNIFORM_CURRICULUM_EXP:-curriculum_lora_uniform_${UNIFORM_VERSION}_stage3_uniform}"
        ;;
    *)
        echo "Error: unsupported UNIFORM_EXP_STYLE=${UNIFORM_EXP_STYLE}. Use uniform_only or legacy_uniform."
        exit 1
        ;;
esac

enabled_count=0
is_enabled "$RUN_ZERO_SHOT" && enabled_count=$((enabled_count + 1))
is_enabled "$RUN_RULE" && enabled_count=$((enabled_count + 1))
is_enabled "$RUN_LOSS" && enabled_count=$((enabled_count + 1))
is_enabled "$RUN_UNIFORM" && enabled_count=$((enabled_count + 1))
is_enabled "$RUN_RANDOM_BUCKET" && enabled_count=$((enabled_count + 1))
is_enabled "$RUN_LLM" && enabled_count=$((enabled_count + 1))
is_enabled "$RUN_MPOC" && enabled_count=$((enabled_count + 1))
if [ "$enabled_count" -eq 0 ]; then
    echo "Error: no methods are enabled. Set at least one RUN_* method flag to true."
    exit 1
fi

normalized_test="$(printf '%s' "$TEST_NAME" | tr '_' '-' | tr '[:upper:]' '[:lower:]')"
case "$normalized_test" in
    val14|val14-benchmark)
        cmd=(bash "${SCRIPT_DIR}/quick_test_val14.sh")
        ;;
    val14-fast|val-fast)
        cmd=(bash "${SCRIPT_DIR}/quick_test_val14-fast.sh")
        ;;
    test14-hard|test14)
        cmd=(bash "${SCRIPT_DIR}/quick_test_test14-hard.sh")
        ;;
    test14-hard-fast|test14-fast|fast)
        cmd=(bash "${SCRIPT_DIR}/quick_test_test14-hard-fast.sh")
        ;;
    interplan10|interplan-10)
        cmd=(bash "${SCRIPT_DIR}/quick_test_interplan.sh" interplan10)
        ;;
    interplan-benchmark|interplan-benchmark-scenario|interplan-benchmark-scenarios|benchmark-scenarios)
        cmd=(bash "${SCRIPT_DIR}/quick_test_interplan.sh" benchmark_scenarios)
        ;;
    *)
        echo "Error: unsupported TEST_NAME=${TEST_NAME}"
        echo "Tests: val14, val14-fast, test14-hard, test14-hard-fast, interplan10, interplan-benchmark"
        exit 1
        ;;
esac

if is_enabled "$DRY_RUN"; then
    echo "Quick-test dispatch:"
    printf '  command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    echo "  methods: zero_shot=${RUN_ZERO_SHOT}, rule=${RUN_RULE}, loss=${RUN_LOSS}, uniform=${RUN_UNIFORM}, random_bucket=${RUN_RANDOM_BUCKET}, llm=${RUN_LLM}, mpoc=${RUN_MPOC}"
    echo "  versions: rule=${RULE_CURRICULUM_VERSION}, loss=${LOSS_CURRICULUM_VERSION}, random_bucket=${RANDOM_BUCKET_CURRICULUM_VERSION}, llm=${LLM_CURRICULUM_VERSION}, mpoc=${MPOC_CURRICULUM_VERSION}, uniform=${UNIFORM_CURRICULUM_VERSION}"
    echo "  slugs: rule=${RULE_CURRICULUM_SLUG}, loss=${LOSS_CURRICULUM_SLUG}, random_bucket=${RANDOM_BUCKET_CURRICULUM_SLUG}, llm=${LLM_CURRICULUM_SLUG}, mpoc=${MPOC_CURRICULUM_SLUG}, uniform=${UNIFORM_CURRICULUM_SLUG}"
    echo "  exps: rule=${RULE_CURRICULUM_EXP}, loss=${LOSS_CURRICULUM_EXP}, random_bucket=${RANDOM_BUCKET_CURRICULUM_EXP}, llm=${LLM_CURRICULUM_EXP}, mpoc=${MPOC_CURRICULUM_EXP}, uniform=${UNIFORM_CURRICULUM_EXP}"
    exit 0
fi

exec "${cmd[@]}"
