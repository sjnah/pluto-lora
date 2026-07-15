#!/usr/bin/env bash
# Train and immediately evaluate each enabled PLUTO LoRA method, one seed at a time.
# Order: method 1 seed range -> method 2 seed range -> ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_SUITE="${SCRIPT_DIR}/run_lora_experiment_suite.sh"
EVAL_SUITE="${REPO_ROOT}/scripts/evaluation/run_quick_test_suite.sh"
CONFIG_RESOLVER="${SCRIPT_DIR}/resolve_lora_experiment_config.py"
cd "$REPO_ROOT"

EXPERIMENT_SUITE_CONFIG="${EXPERIMENT_SUITE_CONFIG:-${REPO_ROOT}/config/experiment_suite/flat_lr_comparison_v1.yaml}"
eval "$(python3 "$CONFIG_RESOLVER" --suite "$EXPERIMENT_SUITE_CONFIG" --format shell)"
TRAINING_PROTOCOL_CONFIG="${TRAINING_PROTOCOL_CONFIG:-$CFG_SUITE_TRAINING_PROTOCOL}"
eval "$(python3 "$CONFIG_RESOLVER" \
    --protocol "$TRAINING_PROTOCOL_CONFIG" \
    --method "${REPO_ROOT}/config/curriculum_method/llm.yaml" \
    --format shell)"
TRAINING_PROTOCOL_ID="$CFG_PROTOCOL_ID"
TRAINING_PROTOCOL_SHA256="$CFG_PROTOCOL_SHA256"

# Method switches.
RUN_LLM="${RUN_LLM:-$CFG_SUITE_RUN_LLM}"
RUN_RULE="${RUN_RULE:-$CFG_SUITE_RUN_RULE}"
RUN_LOSS="${RUN_LOSS:-$CFG_SUITE_RUN_LOSS}"
RUN_RANDOM_BUCKET="${RUN_RANDOM_BUCKET:-$CFG_SUITE_RUN_RANDOM}"
RUN_MPOC="${RUN_MPOC:-$CFG_SUITE_RUN_MPOC}"
RUN_UNIFORM="${RUN_UNIFORM:-$CFG_SUITE_RUN_UNIFORM}"

# Inclusive seed range.
SEED_START="${SEED_START:-$CFG_SUITE_SEED_START}"
SEED_END="${SEED_END:-$CFG_SUITE_SEED_END}"

# Benchmark switches. These defaults match the requested three evaluations.
RUN_VAL14="${RUN_VAL14:-$CFG_SUITE_RUN_VAL14}"
RUN_VAL14_FAST="${RUN_VAL14_FAST:-$CFG_SUITE_RUN_VAL14_FAST}"
RUN_TEST14_HARD="${RUN_TEST14_HARD:-$CFG_SUITE_RUN_TEST14_HARD}"
RUN_TEST14_HARD_FAST="${RUN_TEST14_HARD_FAST:-$CFG_SUITE_RUN_TEST14_HARD_FAST}"
RUN_INTERPLAN10="${RUN_INTERPLAN10:-$CFG_SUITE_RUN_INTERPLAN10}"
RUN_INTERPLAN_BENCHMARK="${RUN_INTERPLAN_BENCHMARK:-$CFG_SUITE_RUN_INTERPLAN_BENCHMARK}"

# Versions and shared behavior.
LLM_VERSION="${LLM_VERSION:-$CFG_SUITE_LLM_VERSION}"
RULE_VERSION="${RULE_VERSION:-$CFG_SUITE_RULE_VERSION}"
LOSS_VERSION="${LOSS_VERSION:-$CFG_SUITE_LOSS_VERSION}"
RANDOM_BUCKET_VERSION="${RANDOM_BUCKET_VERSION:-$CFG_SUITE_RANDOM_VERSION}"
MPOC_VERSION="${MPOC_VERSION:-$CFG_SUITE_MPOC_VERSION}"
UNIFORM_VERSION="${UNIFORM_VERSION:-$CFG_SUITE_UNIFORM_VERSION}"
PERCENTILE_EHU_FINAL_PHASE="${PERCENTILE_EHU_FINAL_PHASE:-phaseC_hard_replay}"
TYPE_ROUTING_MODE="${TYPE_ROUTING_MODE:-$CFG_SUITE_TYPE_ROUTING_MODE}"
FEATURE_CACHE_NAME="${FEATURE_CACHE_NAME:-$CFG_SUITE_FEATURE_CACHE_NAME}"
DRY_RUN="${DRY_RUN:-false}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-$CFG_SUITE_CONTINUE_ON_FAILURE}"
DISABLE_SIMULATION_LOG="${DISABLE_SIMULATION_LOG:-$CFG_SUITE_DISABLE_SIMULATION_LOG}"
# Reuse a completed final-phase checkpoint for the same method/version/seed.
# Set false to force training even when that checkpoint already exists.
SKIP_TRAINING_IF_CHECKPOINT_EXISTS="${SKIP_TRAINING_IF_CHECKPOINT_EXISTS:-$CFG_SUITE_SKIP_TRAINING_IF_CHECKPOINT_EXISTS}"

is_enabled() {
    case "$1" in true|TRUE|True|1|yes|YES|Yes|on|ON|On) return 0 ;; *) return 1 ;; esac
}

if ! [[ "$SEED_START" =~ ^[0-9]+$ && "$SEED_END" =~ ^[0-9]+$ ]] || [ "$SEED_START" -gt "$SEED_END" ]; then
    echo "Error: SEED_START and SEED_END must be non-negative integers with SEED_START <= SEED_END." >&2
    exit 1
fi

enabled_methods=0
for flag in "$RUN_LLM" "$RUN_RULE" "$RUN_LOSS" "$RUN_RANDOM_BUCKET" "$RUN_MPOC" "$RUN_UNIFORM"; do
    is_enabled "$flag" && enabled_methods=$((enabled_methods + 1))
done
if [ "$enabled_methods" -eq 0 ]; then
    echo "Error: no training methods are enabled." >&2
    exit 1
fi

find_checkpoint_for_experiment() {
    local final_exp="$1"
    local required_seed="${2:-}"
    local required_protocol_id="${3:-$TRAINING_PROTOCOL_ID}"
    local required_protocol_sha256="${4:-$TRAINING_PROTOCOL_SHA256}"
    local required_execution_mode="${5:-}"
    local exp_dir parent_dir candidate config_file config_seed protocol_identity
    local actual_protocol_id actual_protocol_sha256 actual_execution_mode

    while IFS= read -r exp_dir; do
        [ -n "$exp_dir" ] || continue
        parent_dir="$(dirname "$(dirname "$exp_dir")")"
        config_file="${parent_dir}/.hydra/config.yaml"
        if [ -n "$required_seed" ]; then
            config_seed="$(sed -n 's/^seed:[[:space:]]*//p' "$config_file" 2>/dev/null | head -n1)"
            [ "$config_seed" = "$required_seed" ] || continue
        fi
        protocol_identity="$(python3 - "$config_file" <<'PY' 2>/dev/null || true
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
lora = payload.get("lora") or {}
print(
    f"{lora.get('training_protocol_id', '')}|"
    f"{lora.get('training_protocol_sha256', '')}|"
    f"{lora.get('execution_mode', '')}"
)
PY
)"
        IFS='|' read -r actual_protocol_id actual_protocol_sha256 actual_execution_mode \
            <<< "$protocol_identity"
        [ "$actual_protocol_id" = "$required_protocol_id" ] || continue
        [ "$actual_protocol_sha256" = "$required_protocol_sha256" ] || continue
        if [ -n "$required_execution_mode" ]; then
            [ "$actual_execution_mode" = "$required_execution_mode" ] || continue
        fi
        for candidate in \
            "${exp_dir}/lora_checkpoints/merged_final.ckpt" \
            "${exp_dir}/checkpoints/last.ckpt" \
            "${parent_dir}/lora_checkpoints/merged_final.ckpt" \
            "${parent_dir}/checkpoints/last.ckpt"; do
            if [ -f "$candidate" ]; then
                case "$candidate" in
                    /*) printf '%s|%s\n' "$candidate" "$final_exp" ;;
                    *) printf '%s/%s|%s\n' "$REPO_ROOT" "$candidate" "$final_exp" ;;
                esac
                return 0
            fi
        done
    done < <(find outputs -type d -name "$final_exp" 2>/dev/null | sort -r)

    return 1
}

find_completed_training_checkpoint() {
    local final_exp="$1"
    local legacy_exp="${2:-}"
    local seed="${3:-}"
    local required_execution_mode="${4:-}"

    find_checkpoint_for_experiment \
        "$final_exp" "$seed" "$TRAINING_PROTOCOL_ID" \
        "$TRAINING_PROTOCOL_SHA256" "$required_execution_mode" && return 0
    if [ -n "$legacy_exp" ]; then
        # Compatibility for early seeded-wrapper runs whose type-routing
        # experiment name accidentally omitted `_seedN`.
        find_checkpoint_for_experiment \
            "$legacy_exp" "$seed" "$TRAINING_PROTOCOL_ID" \
            "$TRAINING_PROTOCOL_SHA256" "$required_execution_mode" && return 0
    fi
    return 1
}

run_method_seed() {
    local method="$1"
    local version="$2"
    local seed="$3"
    local seed_tag="seed${seed}"
    local base_exp slug final_exp legacy_final_exp=""

    case "$method" in
        llm)
            if [ "$TYPE_ROUTING_MODE" = "on" ] || [ "$TYPE_ROUTING_MODE" = "enabled" ]; then
                base_exp="curriculum_lora_llm_percentile_ehu_${version}_${TRAINING_PROTOCOL_ID}_type_on_${seed_tag}"
                slug="curriculum_llm_percentile_ehu_${version}_${TRAINING_PROTOCOL_ID}_type_on_${seed_tag}"
            else
                base_exp="curriculum_lora_llm_percentile_ehu_${version}_${TRAINING_PROTOCOL_ID}_${seed_tag}"
                slug="curriculum_llm_percentile_ehu_${version}_${TRAINING_PROTOCOL_ID}_${seed_tag}"
            fi
            ;;
        rule|loss|mpoc)
            base_exp="curriculum_lora_${method}_percentile_ehu_${version}_${TRAINING_PROTOCOL_ID}_${seed_tag}"
            slug="curriculum_${method}_percentile_ehu_${version}_${TRAINING_PROTOCOL_ID}_${seed_tag}"
            ;;
        random)
            base_exp="curriculum_lora_random_percentile_ehu_${version}_${TRAINING_PROTOCOL_ID}_${seed_tag}"
            slug="curriculum_randombucket_percentile_ehu_${version}_${TRAINING_PROTOCOL_ID}_${seed_tag}"
            ;;
        uniform)
            base_exp="curriculum_lora_uniform_only_${version}_${TRAINING_PROTOCOL_ID}_${seed_tag}"
            slug="curriculum_uniform_${version}_${TRAINING_PROTOCOL_ID}_${seed_tag}"
            ;;
    esac

    if [ "$method" = "uniform" ]; then
        final_exp="${base_exp}_stage3_uniform"
    else
        final_exp="${base_exp}_${PERCENTILE_EHU_FINAL_PHASE}"
    fi

    echo ""
    echo "============================================================"
    echo "Train -> test: method=${method}, version=${version}, seed=${seed}"
    echo "Experiment: ${base_exp}"
    echo "Evaluation slug: ${slug}"
    if [ "$method" = "uniform" ]; then
        echo "Training contract: continuous 12 epochs; no curriculum phase or optimizer reset"
    fi
    echo "============================================================"

    local upper_method
    upper_method="$(printf '%s' "$method" | tr '[:lower:]' '[:upper:]')"
    [ "$method" = random ] && upper_method=RANDOM_BUCKET

    local completed_training="" completed_checkpoint="" evaluation_exp="$final_exp"
    local required_execution_mode=""
    [ "$method" = "uniform" ] && required_execution_mode=continuous_uniform
    if is_enabled "$SKIP_TRAINING_IF_CHECKPOINT_EXISTS"; then
        completed_training="$(find_completed_training_checkpoint \
            "$final_exp" "$legacy_final_exp" "$seed" \
            "$required_execution_mode" || true)"
        if [ -n "$completed_training" ]; then
            completed_checkpoint="${completed_training%%|*}"
            evaluation_exp="${completed_training#*|}"
        fi
    fi

    if [ -n "$completed_checkpoint" ]; then
        echo "Skipping training: completed checkpoint already exists."
        echo "Checkpoint: ${completed_checkpoint}"
    else
        env \
            RUN_LLM=$([ "$method" = llm ] && echo true || echo false) \
            RUN_RULE=$([ "$method" = rule ] && echo true || echo false) \
            RUN_LOSS=$([ "$method" = loss ] && echo true || echo false) \
            RUN_RANDOM_BUCKET=$([ "$method" = random ] && echo true || echo false) \
            RUN_MPOC=$([ "$method" = mpoc ] && echo true || echo false) \
            RUN_UNIFORM=$([ "$method" = uniform ] && echo true || echo false) \
            LLM_VERSION="$LLM_VERSION" RULE_VERSION="$RULE_VERSION" LOSS_VERSION="$LOSS_VERSION" \
            RANDOM_BUCKET_VERSION="$RANDOM_BUCKET_VERSION" MPOC_VERSION="$MPOC_VERSION" UNIFORM_VERSION="$UNIFORM_VERSION" \
            TRAINING_SEED="$seed" SAMPLER_SEED="$seed" \
            EXPERIMENT_SUITE_CONFIG="$EXPERIMENT_SUITE_CONFIG" TRAINING_PROTOCOL_CONFIG="$TRAINING_PROTOCOL_CONFIG" \
            FEATURE_CACHE_NAME="$FEATURE_CACHE_NAME" \
            TYPE_ROUTING_MODE="$TYPE_ROUTING_MODE" RUN_LLM_TYPE_ROUTING_COMPARISON=false \
            LLM_TYPE_ROUTING_OFF_BASE_EXP="$base_exp" LLM_TYPE_ROUTING_ON_BASE_EXP="$base_exp" \
            "${upper_method}_CURRICULUM_BASE_EXP=$base_exp" UNIFORM_CURRICULUM_BASE_EXP="$base_exp" \
            DRY_RUN="$DRY_RUN" CONTINUE_ON_FAILURE="$CONTINUE_ON_FAILURE" \
            bash "$TRAIN_SUITE"
    fi

    env \
        RUN_VAL14="$RUN_VAL14" RUN_VAL14_FAST="$RUN_VAL14_FAST" \
        RUN_TEST14_HARD="$RUN_TEST14_HARD" RUN_TEST14_HARD_FAST="$RUN_TEST14_HARD_FAST" \
        RUN_INTERPLAN10="$RUN_INTERPLAN10" RUN_INTERPLAN_BENCHMARK="$RUN_INTERPLAN_BENCHMARK" \
        RUN_ZERO_SHOT=false RUN_RULE=$([ "$method" = rule ] && echo true || echo false) \
        RUN_LOSS=$([ "$method" = loss ] && echo true || echo false) \
        RUN_UNIFORM=$([ "$method" = uniform ] && echo true || echo false) \
        RUN_RANDOM_BUCKET=$([ "$method" = random ] && echo true || echo false) \
        RUN_LLM=$([ "$method" = llm ] && echo true || echo false) \
        RUN_MPOC=$([ "$method" = mpoc ] && echo true || echo false) \
        LLM_VERSION="$LLM_VERSION" RULE_VERSION="$RULE_VERSION" LOSS_VERSION="$LOSS_VERSION" \
        RANDOM_BUCKET_VERSION="$RANDOM_BUCKET_VERSION" MPOC_VERSION="$MPOC_VERSION" UNIFORM_VERSION="$UNIFORM_VERSION" \
        "${upper_method}_CURRICULUM_EXP=$evaluation_exp" "${upper_method}_CURRICULUM_SLUG=$slug" \
        TYPE_ROUTING_MODE="$TYPE_ROUTING_MODE" RUN_LLM_TYPE_ROUTING_COMPARISON=false \
        EXPERIMENT_SUITE_CONFIG="$EXPERIMENT_SUITE_CONFIG" TRAINING_PROTOCOL_CONFIG="$TRAINING_PROTOCOL_CONFIG" \
        TRAINING_PROTOCOL_ID="$TRAINING_PROTOCOL_ID" \
        DISABLE_SIMULATION_LOG="$DISABLE_SIMULATION_LOG" \
        DRY_RUN="$DRY_RUN" CONTINUE_ON_FAILURE="$CONTINUE_ON_FAILURE" \
        bash "$EVAL_SUITE"
}

for method_spec in \
    "llm:$LLM_VERSION:$RUN_LLM" \
    "rule:$RULE_VERSION:$RUN_RULE" \
    "loss:$LOSS_VERSION:$RUN_LOSS" \
    "random:$RANDOM_BUCKET_VERSION:$RUN_RANDOM_BUCKET" \
    "mpoc:$MPOC_VERSION:$RUN_MPOC" \
    "uniform:$UNIFORM_VERSION:$RUN_UNIFORM"; do
    IFS=: read -r method version enabled <<< "$method_spec"
    if is_enabled "$enabled"; then
        for ((seed=SEED_START; seed<=SEED_END; seed++)); do
            run_method_seed "$method" "$version" "$seed"
        done
    fi
done

echo ""
echo "All enabled method/seed training and evaluation runs finished."
