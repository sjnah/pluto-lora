#!/usr/bin/env bash
# Evaluate already-trained curriculum phase A/B/C LoRA checkpoints without
# retraining. Uniform FT is not a staged curriculum in the current pipeline, so
# its default evaluation target is the final Phase-C-equivalent checkpoint only.
#
# Default diagnostic matrix:
#   reference:   zero-shot (once per benchmark)
#   methods:     uniform, random, llm
#   seeds:       1, 2, 3
#   phases:      A, B, C
#   checkpoints: standard, EMA
#   benchmark:   test14-hard-fast
#
# The script creates temporary checkpoint views under outputs/, gives every
# phase/variant an independent result slug, and removes the views on exit. It
# never modifies or replaces the original training checkpoints.
#
# Examples:
#   bash scripts/evaluation/run_phase_checkpoint_ablation.sh
#   DRY_RUN=true bash scripts/evaluation/run_phase_checkpoint_ablation.sh
#   PREFLIGHT_ONLY=true bash scripts/evaluation/run_phase_checkpoint_ablation.sh
#   UNIFORM_VERSION=v1.16 RANDOM_BUCKET_VERSION=v1.16 PREFLIGHT_ONLY=true \
#     METHODS=uniform,random bash scripts/evaluation/run_phase_checkpoint_ablation.sh
#   METHODS=uniform SEEDS=1 PHASES=A,B,C CHECKPOINT_VARIANTS=standard \
#     bash scripts/evaluation/run_phase_checkpoint_ablation.sh
#   METHODS=uniform,random,llm SEEDS=1,2,3,4,5 \
#     CHECKPOINT_VARIANTS=standard BENCHMARKS=val14-fast \
#     bash scripts/evaluation/run_phase_checkpoint_ablation.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_RESOLVER="${REPO_ROOT}/scripts/training/resolve_lora_experiment_config.py"
QUICK_TEST_UNIFIED="${SCRIPT_DIR}/quick_test_unified.sh"

cd "$REPO_ROOT"

EXPERIMENT_SUITE_CONFIG="${EXPERIMENT_SUITE_CONFIG:-${REPO_ROOT}/config/experiment_suite/flat_lr_comparison_v1.yaml}"
eval "$(python3 "$CONFIG_RESOLVER" --suite "$EXPERIMENT_SUITE_CONFIG" --format shell)"
TRAINING_PROTOCOL_CONFIG="${TRAINING_PROTOCOL_CONFIG:-$CFG_SUITE_TRAINING_PROTOCOL}"

METHODS="${METHODS:-uniform,random,llm}"
SEEDS="${SEEDS:-1,2,3}"
PHASES="${PHASES:-A,B,C}"
CHECKPOINT_VARIANTS="${CHECKPOINT_VARIANTS:-standard,ema}"
BENCHMARKS="${BENCHMARKS:-test14-hard-fast}"
RUN_ZERO_SHOT_REFERENCE="${RUN_ZERO_SHOT_REFERENCE:-true}"
DRY_RUN="${DRY_RUN:-false}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-false}"
COLLECT_RESULTS="${COLLECT_RESULTS:-true}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-false}"
DISABLE_SIMULATION_LOG="${DISABLE_SIMULATION_LOG:-true}"
TYPE_ROUTING_MODE="${TYPE_ROUTING_MODE:-$CFG_SUITE_TYPE_ROUTING_MODE}"
SKIP_UNIFORM_INTERMEDIATE_PHASES="${SKIP_UNIFORM_INTERMEDIATE_PHASES:-true}"

LLM_VERSION="${LLM_VERSION:-$CFG_SUITE_LLM_VERSION}"
RANDOM_BUCKET_VERSION="${RANDOM_BUCKET_VERSION:-$CFG_SUITE_RANDOM_VERSION}"
UNIFORM_VERSION="${UNIFORM_VERSION:-$CFG_SUITE_UNIFORM_VERSION}"
MPOC_VERSION="${MPOC_VERSION:-$CFG_SUITE_MPOC_VERSION}"

TEMP_VIEW_ROOT="${REPO_ROOT}/outputs/.phase_checkpoint_eval_views_$$"
RESULT_SLUGS=()
FAILED_JOBS=0
FAILURE_MESSAGES=()

cleanup() {
    rm -rf "$TEMP_VIEW_ROOT"
}
trap cleanup EXIT INT TERM

is_enabled() {
    case "$1" in
        true|TRUE|True|1|yes|YES|Yes|on|ON|On) return 0 ;;
        *) return 1 ;;
    esac
}

normalize_csv() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -d ' '
}

add_result_slug() {
    local candidate="$1"
    local existing
    for existing in "${RESULT_SLUGS[@]}"; do
        [ "$existing" = "$candidate" ] && return 0
    done
    RESULT_SLUGS+=("$candidate")
}

record_failure() {
    local message="$1"
    FAILED_JOBS=$((FAILED_JOBS + 1))
    FAILURE_MESSAGES+=("$message")
    echo "Error: $message" >&2

    if ! is_enabled "$PREFLIGHT_ONLY" && ! is_enabled "$CONTINUE_ON_FAILURE"; then
        exit 1
    fi
}

latest_experiment_dir() {
    local experiment_name="$1"
    find outputs -type d -name "$experiment_name" -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr \
        | head -n1 \
        | cut -d' ' -f2-
}

phase_key_lower() {
    case "$1" in
        A|a) printf 'a' ;;
        B|b) printf 'b' ;;
        C|c) printf 'c' ;;
        *) return 1 ;;
    esac
}

checkpoint_filename() {
    case "$1" in
        standard) printf 'merged_final.ckpt' ;;
        ema) printf 'merged_final_ema.ckpt' ;;
        *) return 1 ;;
    esac
}

method_version() {
    case "$1" in
        uniform) printf '%s' "$UNIFORM_VERSION" ;;
        random) printf '%s' "$RANDOM_BUCKET_VERSION" ;;
        llm) printf '%s' "$LLM_VERSION" ;;
        mpoc) printf '%s' "$MPOC_VERSION" ;;
        *) return 1 ;;
    esac
}

method_run_flag() {
    case "$1" in
        uniform) printf 'RUN_UNIFORM' ;;
        random) printf 'RUN_RANDOM_BUCKET' ;;
        llm) printf 'RUN_LLM' ;;
        mpoc) printf 'RUN_MPOC' ;;
        *) return 1 ;;
    esac
}

method_exp_override() {
    case "$1" in
        uniform) printf 'UNIFORM_CURRICULUM_EXP' ;;
        random) printf 'RANDOM_BUCKET_CURRICULUM_EXP' ;;
        llm) printf 'LLM_CURRICULUM_EXP' ;;
        mpoc) printf 'MPOC_CURRICULUM_EXP' ;;
        *) return 1 ;;
    esac
}

method_slug_override() {
    case "$1" in
        uniform) printf 'UNIFORM_CURRICULUM_SLUG' ;;
        random) printf 'RANDOM_BUCKET_CURRICULUM_SLUG' ;;
        llm) printf 'LLM_CURRICULUM_SLUG' ;;
        mpoc) printf 'MPOC_CURRICULUM_SLUG' ;;
        *) return 1 ;;
    esac
}

resolve_method_contract() {
    local method="$1"
    local method_config="${REPO_ROOT}/config/curriculum_method/${method}.yaml"
    eval "$(python3 "$CONFIG_RESOLVER" \
        --protocol "$TRAINING_PROTOCOL_CONFIG" \
        --method "$method_config" \
        --format shell)"
}

source_experiment_name() {
    local method="$1"
    local version="$2"
    local seed="$3"
    local phase="$4"
    local base

    case "$method" in
        uniform)
            base="curriculum_lora_uniform_only_${version}_${CFG_PROTOCOL_ID}_seed${seed}"
            ;;
        llm)
            if [ "$TYPE_ROUTING_MODE" = "on" ] || [ "$TYPE_ROUTING_MODE" = "enabled" ]; then
                base="curriculum_lora_llm_percentile_ehu_${version}_${CFG_PROTOCOL_ID}_type_on_seed${seed}"
            else
                base="curriculum_lora_llm_percentile_ehu_${version}_${CFG_PROTOCOL_ID}_seed${seed}"
            fi
            ;;
        random|mpoc)
            base="curriculum_lora_${method}_percentile_ehu_${version}_${CFG_PROTOCOL_ID}_seed${seed}"
            ;;
        *)
            return 1
            ;;
    esac

    if [ "$method" = "uniform" ]; then
        case "$phase" in
            a) printf '%s_%s' "$base" "$CFG_PHASE_A_NAME" ;;
            b) printf '%s_%s' "$base" "$CFG_PHASE_B_NAME" ;;
            c) printf '%s_%s' "$base" "$CFG_PHASE_C_NAME" ;;
        esac
    else
        case "$phase" in
            a) printf '%s_phaseA_%s' "$base" "$CFG_PHASE_A_NAME" ;;
            b) printf '%s_phaseB_%s' "$base" "$CFG_PHASE_B_NAME" ;;
            c) printf '%s_phaseC_%s' "$base" "$CFG_PHASE_C_NAME" ;;
        esac
    fi
}

result_slug() {
    local method="$1"
    local version="$2"
    local seed="$3"
    local phase="$4"
    local variant="$5"
    local method_part="$method"

    [ "$method" = "random" ] && method_part="randombucket"
    if [ "$method" = "uniform" ]; then
        printf 'curriculum_uniform_%s_%s_phase%s_%s_seed%s' \
            "$version" "$CFG_PROTOCOL_ID" "$phase" "$variant" "$seed"
    else
        printf 'curriculum_%s_percentile_ehu_%s_%s_phase%s_%s_seed%s' \
            "$method_part" "$version" "$CFG_PROTOCOL_ID" "$phase" "$variant" "$seed"
    fi
}

should_skip_phase() {
    local method="$1"
    local phase="$2"

    if [ "$method" = "uniform" ] && is_enabled "$SKIP_UNIFORM_INTERMEDIATE_PHASES" \
        && [ "$phase" != "c" ]; then
        return 0
    fi
    return 1
}

run_zero_shot_reference() {
    local benchmark="$1"
    local checkpoint="${REPO_ROOT}/checkpoints/pluto_1M_aux_cil.ckpt"

    echo ""
    echo "============================================================"
    echo "Zero-shot reference evaluation"
    echo "  benchmark:  $benchmark"
    echo "  result slug: zeroshot"
    echo "============================================================"

    if ! is_enabled "$DRY_RUN" && [ ! -f "$checkpoint" ]; then
        record_failure "zero-shot checkpoint not found: $checkpoint"
        return 0
    fi

    if is_enabled "$PREFLIGHT_ONLY"; then
        echo "  checkpoint: $checkpoint"
        echo "  preflight:  passed"
        return 0
    fi

    set +e
    env \
        RUN_ZERO_SHOT=true RUN_RULE=false RUN_LOSS=false \
        RUN_UNIFORM=false RUN_RANDOM_BUCKET=false RUN_LLM=false RUN_MPOC=false \
        DISABLE_SIMULATION_LOG="$DISABLE_SIMULATION_LOG" \
        DRY_RUN="$DRY_RUN" \
        bash "$QUICK_TEST_UNIFIED" "$benchmark"
    local status=$?
    set -e

    if [ "$status" -ne 0 ]; then
        record_failure "zero-shot evaluation failed with status $status ($benchmark)"
        return 0
    fi
    add_result_slug "zeroshot"
}

run_one() {
    local method="$1"
    local seed="$2"
    local phase_upper="$3"
    local variant="$4"
    local benchmark="$5"
    local phase version source_exp source_dir filename source_ckpt slug view_exp view_dir
    local run_flag exp_override slug_override

    phase="$(phase_key_lower "$phase_upper")" || {
        echo "Error: unsupported phase: $phase_upper (use A, B, or C)" >&2
        exit 1
    }
    if should_skip_phase "$method" "$phase"; then
        echo ""
        echo "Skipping Uniform FT intermediate request: phase $(printf '%s' "$phase" | tr '[:lower:]' '[:upper:]')"
        echo "  reason: uniform FT is not staged by curriculum phase; evaluating final phase C only"
        return 0
    fi

    filename="$(checkpoint_filename "$variant")" || {
        echo "Error: unsupported checkpoint variant: $variant (use standard or ema)" >&2
        exit 1
    }

    resolve_method_contract "$method"
    version="$(method_version "$method")" || {
        echo "Error: unsupported method: $method (use uniform, random, llm, or mpoc)" >&2
        exit 1
    }
    source_exp="$(source_experiment_name "$method" "$version" "$seed" "$phase")"
    slug="$(result_slug "$method" "$version" "$seed" "$(printf '%s' "$phase" | tr '[:lower:]' '[:upper:]')" "$variant")"
    view_exp="phase_checkpoint_view_${slug}"
    view_dir="${TEMP_VIEW_ROOT}/outputs/${view_exp}/lora_checkpoints"

    echo ""
    echo "============================================================"
    echo "Phase checkpoint evaluation"
    echo "  method:     $method"
    echo "  version:    $version"
    echo "  seed:       $seed"
    echo "  phase:      $(printf '%s' "$phase" | tr '[:lower:]' '[:upper:]')"
    echo "  variant:    $variant ($filename)"
    echo "  benchmark:  $benchmark"
    echo "  source exp: $source_exp"
    echo "  result slug: $slug"
    echo "============================================================"

    if ! is_enabled "$DRY_RUN"; then
        source_dir="$(latest_experiment_dir "$source_exp")"
        if [ -z "$source_dir" ]; then
            record_failure "training output not found: $source_exp"
            return 0
        fi
        source_ckpt="${source_dir}/lora_checkpoints/${filename}"
        if [ ! -f "$source_ckpt" ]; then
            record_failure "checkpoint not found: $source_ckpt"
            return 0
        fi
        mkdir -p "$view_dir"
        ln -sfn "$(realpath "$source_ckpt")" "${view_dir}/merged_final.ckpt"
        echo "  checkpoint: $source_ckpt"
    fi

    if is_enabled "$PREFLIGHT_ONLY"; then
        echo "  preflight:  passed"
        return 0
    fi

    run_flag="$(method_run_flag "$method")"
    exp_override="$(method_exp_override "$method")"
    slug_override="$(method_slug_override "$method")"

    set +e
    env \
        RUN_ZERO_SHOT=false RUN_RULE=false RUN_LOSS=false \
        RUN_UNIFORM=false RUN_RANDOM_BUCKET=false RUN_LLM=false RUN_MPOC=false \
        "$run_flag=true" \
        "$exp_override=$view_exp" \
        "$slug_override=$slug" \
        LLM_VERSION="$LLM_VERSION" \
        RANDOM_BUCKET_VERSION="$RANDOM_BUCKET_VERSION" \
        UNIFORM_VERSION="$UNIFORM_VERSION" \
        MPOC_VERSION="$MPOC_VERSION" \
        TRAINING_PROTOCOL_ID="$CFG_PROTOCOL_ID" \
        TYPE_ROUTING_MODE="$TYPE_ROUTING_MODE" \
        RUN_LLM_TYPE_ROUTING_COMPARISON=false \
        DISABLE_SIMULATION_LOG="$DISABLE_SIMULATION_LOG" \
        DRY_RUN="$DRY_RUN" \
        bash "$QUICK_TEST_UNIFIED" "$benchmark"
    local status=$?
    set -e

    if [ "$status" -ne 0 ]; then
        record_failure "evaluation failed with status $status: $slug ($benchmark)"
        return 0
    fi
    add_result_slug "$slug"
}

METHODS="$(normalize_csv "$METHODS")"
SEEDS="$(normalize_csv "$SEEDS")"
PHASES="$(normalize_csv "$PHASES")"
CHECKPOINT_VARIANTS="$(normalize_csv "$CHECKPOINT_VARIANTS")"
BENCHMARKS="$(normalize_csv "$BENCHMARKS")"

if is_enabled "$DRY_RUN" && is_enabled "$PREFLIGHT_ONLY"; then
    echo "Error: DRY_RUN and PREFLIGHT_ONLY cannot both be enabled." >&2
    exit 1
fi

if ! is_enabled "$DRY_RUN" && ! is_enabled "$PREFLIGHT_ONLY"; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/scripts/env_bootstrap.sh"
fi

IFS=',' read -r -a method_list <<< "$METHODS"
IFS=',' read -r -a seed_list <<< "$SEEDS"
IFS=',' read -r -a phase_list <<< "$PHASES"
IFS=',' read -r -a variant_list <<< "$CHECKPOINT_VARIANTS"
IFS=',' read -r -a benchmark_list <<< "$BENCHMARKS"

for seed in "${seed_list[@]}"; do
    if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
        echo "Error: invalid seed: $seed" >&2
        exit 1
    fi
done

total_jobs=0
for method in "${method_list[@]}"; do
    for phase in "${phase_list[@]}"; do
        phase_lower="$(phase_key_lower "$phase")" || {
            echo "Error: unsupported phase: $phase (use A, B, or C)" >&2
            exit 1
        }
        if should_skip_phase "$method" "$phase_lower"; then
            continue
        fi
        total_jobs=$((total_jobs + ${#seed_list[@]} * ${#variant_list[@]} * ${#benchmark_list[@]}))
    done
done
if is_enabled "$RUN_ZERO_SHOT_REFERENCE"; then
    total_jobs=$((total_jobs + ${#benchmark_list[@]}))
fi
if is_enabled "$SKIP_UNIFORM_INTERMEDIATE_PHASES"; then
    uniform_phase_policy="skip A/B"
else
    uniform_phase_policy="evaluate requested phases"
fi
echo "Phase checkpoint ablation"
echo "Suite:       $CFG_SUITE_ID"
echo "Protocol:    $TRAINING_PROTOCOL_CONFIG"
echo "Methods:     $METHODS"
echo "Seeds:       $SEEDS"
echo "Phases:      $PHASES"
echo "Variants:    $CHECKPOINT_VARIANTS"
echo "Benchmarks:  $BENCHMARKS"
echo "Zero-shot:   $RUN_ZERO_SHOT_REFERENCE"
echo "Total jobs:  $total_jobs"
echo "Retraining:  disabled"
echo "Preflight:   $PREFLIGHT_ONLY"
echo "Collection:  $COLLECT_RESULTS"
echo "Uniform:     $uniform_phase_policy"

if is_enabled "$RUN_ZERO_SHOT_REFERENCE"; then
    for benchmark in "${benchmark_list[@]}"; do
        run_zero_shot_reference "$benchmark"
    done
fi

for method in "${method_list[@]}"; do
    for seed in "${seed_list[@]}"; do
        for phase in "${phase_list[@]}"; do
            phase_lower="$(phase_key_lower "$phase")"
            if should_skip_phase "$method" "$phase_lower"; then
                echo ""
                echo "Skipping Uniform FT intermediate request: phase $(printf '%s' "$phase_lower" | tr '[:lower:]' '[:upper:]')"
                echo "  reason: uniform FT is not staged by curriculum phase; evaluating final phase C only"
                continue
            fi
            for variant in "${variant_list[@]}"; do
                for benchmark in "${benchmark_list[@]}"; do
                    run_one "$method" "$seed" "$phase" "$variant" "$benchmark"
                done
            done
        done
    done
done

if ! is_enabled "$DRY_RUN" && ! is_enabled "$PREFLIGHT_ONLY" \
    && is_enabled "$COLLECT_RESULTS" && [ "${#RESULT_SLUGS[@]}" -gt 0 ]; then
    collect_methods="$(IFS=,; echo "${RESULT_SLUGS[*]}")"
    echo ""
    echo "Collecting phase/checkpoint comparison summary..."
    python "${REPO_ROOT}/scripts/evaluation/collect_quick_test_results.py" \
        --tests "$BENCHMARKS" \
        --methods "$collect_methods" \
        --detail || echo "Warning: final phase/checkpoint result collection failed."
fi

echo ""
if [ "$FAILED_JOBS" -gt 0 ]; then
    echo "Phase checkpoint ablation finished with $FAILED_JOBS failed job(s)." >&2
    for message in "${FAILURE_MESSAGES[@]}"; do
        echo "  - $message" >&2
    done
    exit 1
fi

if is_enabled "$PREFLIGHT_ONLY"; then
    echo "All requested phase checkpoints are available."
else
    echo "All phase checkpoint evaluations finished."
fi
