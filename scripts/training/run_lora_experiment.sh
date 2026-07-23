#!/usr/bin/env bash
# Canonical three-phase PLUTO LoRA experiment runner.
# Common optimization values come only from TRAINING_PROTOCOL_CONFIG; method
# files own only data/sampling behavior.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
PLUTO_FILTER_DIR="${REPO_ROOT}/config/scenario_filter"
CONFIG_RESOLVER="${SCRIPT_DIR}/resolve_lora_experiment_config.py"
FILTER_VALIDATOR="${SCRIPT_DIR}/validate_curriculum_filters.py"
export PLUTO_CONFIG_ROOT="${PLUTO_CONFIG_ROOT:-${REPO_ROOT}/config}"
# Separate Hydra metadata even when independent canonical runners start at the
# same instant. The timestamp still distinguishes this runner's three phases.
export PLUTO_TRAINING_RUN_SUFFIX="${PLUTO_TRAINING_RUN_SUFFIX:-$$}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/python_runtime.sh"

METHOD="${METHOD:-llm}"
TRAINING_PROTOCOL_CONFIG="${TRAINING_PROTOCOL_CONFIG:-${REPO_ROOT}/config/training_protocol/flat_area_matched_v1.yaml}"
if [ -z "${METHOD_CONFIG:-}" ]; then
    if [ "$METHOD" = llm ]; then
        METHOD_CONFIG="${REPO_ROOT}/config/curriculum_method/llm_capped.yaml"
    else
        METHOD_CONFIG="${REPO_ROOT}/config/curriculum_method/${METHOD}.yaml"
    fi
fi

eval "$("$PYTHON_BIN" "$CONFIG_RESOLVER" \
    --protocol "$TRAINING_PROTOCOL_CONFIG" \
    --method "$METHOD_CONFIG" \
    --format shell)"

if [ "$METHOD" != "$CFG_METHOD" ]; then
    echo "Error: METHOD=$METHOD does not match method config $CFG_METHOD_PATH ($CFG_METHOD)" >&2
    exit 1
fi
if [ "$CFG_SCHEDULER_HORIZON_EPOCHS" -ne "$CFG_EPOCHS_PHASE_C" ]; then
    echo "Error: scheduler horizon must equal the final cumulative phase boundary" >&2
    exit 1
fi

PROTOCOL_ID="$CFG_PROTOCOL_ID"
PROTOCOL_SHA256="$CFG_PROTOCOL_SHA256"
METHOD_SHA256="$CFG_METHOD_SHA256"
METHOD_LABEL="${METHOD_LABEL:-$CFG_METHOD_LABEL}"
CURRICULUM_VERSION="${CURRICULUM_VERSION:-unversioned}"
CURRICULUM_ARTIFACT_BUNDLE_ID="${CURRICULUM_ARTIFACT_BUNDLE_ID:-}"
CURRICULUM_ARTIFACT_BUNDLE_MANIFEST_SHA256="${CURRICULUM_ARTIFACT_BUNDLE_MANIFEST_SHA256:-}"
DRY_RUN="${DRY_RUN:-false}"
START_PHASE="${START_PHASE:-auto}"
FRESH_START="${FRESH_START:-false}"
PERCENTILE_SPLIT_SEED="${PERCENTILE_SPLIT_SEED:-42}"
SAMPLER_SEED="${SAMPLER_SEED:-$PERCENTILE_SPLIT_SEED}"
TRAINING_SEED="${TRAINING_SEED:-$SAMPLER_SEED}"

PRETRAINED_CKPT="${PRETRAINED_CKPT:-${REPO_ROOT}/checkpoints/pluto_1M_aux_cil.ckpt}"
LORA_ENABLED="$CFG_LORA_ENABLED"
LORA_RANK="$CFG_LORA_RANK"
LORA_ALPHA="$CFG_LORA_ALPHA"
LORA_DROPOUT="$CFG_LORA_DROPOUT"
LORA_LR="$CFG_LORA_LR"
HEAD_LR="$CFG_HEAD_LR"
LR="$CFG_BASE_LR"
WEIGHT_DECAY="$CFG_WEIGHT_DECAY"
EPOCHS_PHASE_A="$CFG_EPOCHS_PHASE_A"
EPOCHS_PHASE_B="$CFG_EPOCHS_PHASE_B"
EPOCHS_PHASE_C="$CFG_EPOCHS_PHASE_C"
WARMUP_STEPS="$CFG_WARMUP_STEPS"
SCHEDULER_TYPE="$CFG_SCHEDULER_TYPE"
SCHEDULER_TOTAL_STEPS="$CFG_SCHEDULER_TOTAL_STEPS"
SCHEDULER_MIN_LR="$CFG_SCHEDULER_MIN_LR"
RESET_OPTIMIZER_MOMENTS_AT_PHASE_B="$CFG_RESET_AT_PHASE_B"
GRADIENT_CLIP_VAL="$CFG_GRADIENT_CLIP_VAL"
SKIP_NAN_STEPS="$CFG_SKIP_NAN_STEPS"
REMOVE_INVALID_GOALS="$CFG_REMOVE_INVALID_GOALS"
ULTRA_MINIMAL="$CFG_ULTRA_MINIMAL"
BATCH_SIZE="$CFG_BATCH_SIZE"
ACCUMULATE_GRAD_BATCHES="$CFG_ACCUMULATE_GRAD_BATCHES"
NUM_SANITY_VAL_STEPS="$CFG_NUM_SANITY_VAL_STEPS"
PRESERVE_L2SP_ANCHOR_ON_RESUME="$CFG_PRESERVE_L2SP_ANCHOR_ON_RESUME"
PRESERVE_RNG_STATE_ON_RESUME="$CFG_PRESERVE_RNG_STATE_ON_RESUME"

FEATURE_CACHE_NAME="${FEATURE_CACHE_NAME:-}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-${REPO_ROOT}/artifacts/feature_cache}"
FEATURE_CACHE_PATH=""
if [ -n "$FEATURE_CACHE_NAME" ]; then
    if ! [[ "$FEATURE_CACHE_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
        echo "Error: invalid FEATURE_CACHE_NAME: $FEATURE_CACHE_NAME" >&2
        exit 1
    fi
    case "$FEATURE_CACHE_ROOT" in
        /*) ;;
        *) FEATURE_CACHE_ROOT="${REPO_ROOT}/${FEATURE_CACHE_ROOT}" ;;
    esac
    FEATURE_CACHE_PATH="${FEATURE_CACHE_ROOT}/${FEATURE_CACHE_NAME}"
fi

PHASE_A_NAME="$CFG_PHASE_A_NAME"
PHASE_B_NAME="$CFG_PHASE_B_NAME"
PHASE_C_NAME="$CFG_PHASE_C_NAME"
PHASE_A_TARGET_PROPORTIONS="$CFG_PHASE_A_TARGET_PROPORTIONS"
PHASE_B_TARGET_PROPORTIONS="$CFG_PHASE_B_TARGET_PROPORTIONS"
PHASE_C_TARGET_PROPORTIONS="$CFG_PHASE_C_TARGET_PROPORTIONS"
PHASE_A_PACING_SCHEDULE="$CFG_PHASE_A_PACING_SCHEDULE"
PHASE_B_PACING_SCHEDULE="$CFG_PHASE_B_PACING_SCHEDULE"
PHASE_C_PACING_SCHEDULE="$CFG_PHASE_C_PACING_SCHEDULE"
PHASE_POLICY="$CFG_PHASE_POLICY"

is_enabled() {
    case "$1" in
        true|TRUE|True|1|yes|YES|Yes|on|ON|On) return 0 ;;
        *) return 1 ;;
    esac
}

if [ "$CFG_METHOD_MODE" = "uniform" ]; then
    default_base="curriculum_lora_uniform_only_${CURRICULUM_VERSION}_${PROTOCOL_ID}"
else
    default_base="curriculum_lora_${METHOD}_percentile_ehu_${CURRICULUM_VERSION}_${PROTOCOL_ID}"
fi
CURRICULUM_BASE_EXP="${CURRICULUM_BASE_EXP:-$default_base}"
PROTOCOL_SNAPSHOT_PATH="${REPO_ROOT}/artifacts/training_protocols/${CURRICULUM_BASE_EXP}.${PROTOCOL_SHA256:0:12}.${METHOD_SHA256:0:12}.json"

FILTER_PREFIX="${FILTER_PREFIX:-$CFG_FILTER_PREFIX}"
SCENARIO_FILTER_UNIFORM="$CFG_SCENARIO_FILTER_UNIFORM"
SCENARIO_FILTER_EASY="${FILTER_PREFIX}_train_easy"
SCENARIO_FILTER_MEDIUM="${FILTER_PREFIX}_train_medium"
SCENARIO_FILTER_HARD="${FILTER_PREFIX}_train_hard"
CURRICULUM_SPLITS="[$SCENARIO_FILTER_EASY,$SCENARIO_FILTER_MEDIUM,$SCENARIO_FILTER_HARD]"

BUCKETIZATION_MODE="$CFG_BUCKETIZATION_MODE"
BUCKET_CARDINALITY_CONTRACT="$CFG_BUCKET_CARDINALITY_CONTRACT"
BUCKET_MASTER_FILTER="$CFG_BUCKET_MASTER_FILTER"
VALIDATION_FILTER="$CFG_VALIDATION_FILTER"
TIE_BREAK_MODE="${TIE_BREAK_MODE:-$CFG_TIE_BREAK_MODE}"
SAMPLER_MODE="${SAMPLER_MODE:-$CFG_SAMPLER_MODE}"
MAX_REPEAT_PER_SCENARIO="$CFG_MAX_REPEAT_PER_SCENARIO"
HARD_SUBTYPE_BALANCE="$CFG_HARD_SUBTYPE_BALANCE"
CURRICULUM_METHOD="$CFG_CURRICULUM_METHOD"
MAX_REPEAT_PER_NEAR_DUPLICATE_GROUP="$CFG_MAX_REPEAT_PER_NEAR_DUPLICATE_GROUP"
NEAR_DUPLICATE_GROUP_WEIGHTING="$CFG_NEAR_DUPLICATE_GROUP_WEIGHTING"
MAX_CUMULATIVE_EXPOSURE_PER_SCENARIO="$CFG_MAX_CUMULATIVE_EXPOSURE_PER_SCENARIO"
MAX_CUMULATIVE_EXPOSURE_PER_NEAR_DUPLICATE_GROUP="$CFG_MAX_CUMULATIVE_EXPOSURE_PER_NEAR_DUPLICATE_GROUP"

TYPE_ROUTING_MODE="${TYPE_ROUTING_MODE:-$CFG_TYPE_ROUTING_DEFAULT_MODE}"
TYPE_ROUTING_METADATA_PATH="${TYPE_ROUTING_METADATA_PATH:-$CFG_TYPE_ROUTING_METADATA_PATH}"
TYPE_ROUTING_STRENGTH="$CFG_TYPE_ROUTING_STRENGTH"
TYPE_ROUTING_METADATA_SHA256=""
if [ "$CFG_TYPE_ROUTING_SUPPORTED" = "true" ]; then
    case "$TYPE_ROUTING_MODE" in
        off|observe_only)
            DEMONSTRATION_TYPE_MODE=observe_only
            DEMONSTRATION_TYPE_METADATA_PATH=null
            ;;
        on|enabled)
            DEMONSTRATION_TYPE_MODE=enabled
            DEMONSTRATION_TYPE_METADATA_PATH="$TYPE_ROUTING_METADATA_PATH"
            SAMPLER_MODE="$CFG_TYPE_ROUTING_ENABLED_SAMPLER_MODE"
            ;;
        *)
            echo "Error: TYPE_ROUTING_MODE must be off/on, got: $TYPE_ROUTING_MODE" >&2
            exit 1
            ;;
    esac
else
    case "$TYPE_ROUTING_MODE" in off|observe_only) ;; *)
        echo "Error: type routing is not supported for method $METHOD" >&2
        exit 1
    esac
    DEMONSTRATION_TYPE_MODE=observe_only
    DEMONSTRATION_TYPE_METADATA_PATH=null
fi

if [ "$CFG_PERSISTENT_EXPOSURE" = "true" ]; then
    CUMULATIVE_EXPOSURE_STATE_PATH="${REPO_ROOT}/artifacts/curriculum_sampling/${CURRICULUM_BASE_EXP}_cumulative_exposure.json"
else
    CUMULATIVE_EXPOSURE_STATE_PATH=null
fi

freeze_type_routing_metadata() {
    if [ "$DEMONSTRATION_TYPE_MODE" != "enabled" ]; then
        return 0
    fi
    local source_path="$TYPE_ROUTING_METADATA_PATH"
    local snapshot_path="${TYPE_ROUTING_SNAPSHOT_PATH:-${REPO_ROOT}/artifacts/curriculum_sampling/${CURRICULUM_BASE_EXP}_type_metadata.csv}"
    local temporary_path="${snapshot_path}.tmp.$$"
    mkdir -p "$(dirname "$snapshot_path")"
    if [ -f "$snapshot_path" ]; then
        if [ -f "$source_path" ] && ! cmp -s "$source_path" "$snapshot_path"; then
            echo "Error: type-routing metadata changed for existing experiment snapshot." >&2
            exit 1
        fi
    elif [ -f "$source_path" ]; then
        cp "$source_path" "$temporary_path"
        mv "$temporary_path" "$snapshot_path"
    else
        echo "Error: type-routing metadata and frozen snapshot are both missing." >&2
        exit 1
    fi
    DEMONSTRATION_TYPE_METADATA_PATH="$snapshot_path"
    TYPE_ROUTING_METADATA_SHA256="$(sha256sum "$snapshot_path" | awk '{print $1}')"
}

verify_type_routing_metadata_snapshot() {
    if [ "$DEMONSTRATION_TYPE_MODE" != "enabled" ]; then
        return 0
    fi
    if [ ! -f "$DEMONSTRATION_TYPE_METADATA_PATH" ]; then
        echo "Error: frozen type-routing metadata disappeared: $DEMONSTRATION_TYPE_METADATA_PATH" >&2
        exit 1
    fi
    local actual_sha256
    actual_sha256="$(sha256sum "$DEMONSTRATION_TYPE_METADATA_PATH" | awk '{print $1}')"
    if [ "$actual_sha256" != "$TYPE_ROUTING_METADATA_SHA256" ]; then
        echo "Error: frozen type-routing metadata checksum changed." >&2
        exit 1
    fi
}

ensure_method_filters() {
    if [ -n "$VALIDATION_FILTER" ] && [ ! -f "${PLUTO_FILTER_DIR}/${VALIDATION_FILTER}.yaml" ]; then
        echo "Missing validation scenario filter: ${VALIDATION_FILTER}.yaml" >&2
        exit 1
    fi
    if [ "$CFG_METHOD_MODE" = "uniform" ]; then
        if [ ! -f "${PLUTO_FILTER_DIR}/${SCENARIO_FILTER_UNIFORM}.yaml" ]; then
            echo "Missing Uniform scenario filter: ${SCENARIO_FILTER_UNIFORM}.yaml" >&2
            exit 1
        fi
        return 0
    fi
    local filter_name
    for filter_name in "$SCENARIO_FILTER_EASY" "$SCENARIO_FILTER_MEDIUM" "$SCENARIO_FILTER_HARD"; do
        if [ ! -f "${PLUTO_FILTER_DIR}/${filter_name}.yaml" ]; then
            echo "Missing percentile scenario filter: ${filter_name}.yaml" >&2
            exit 1
        fi
    done
    if [ "$BUCKET_CARDINALITY_CONTRACT" != "none" ]; then
        "$PYTHON_BIN" "$FILTER_VALIDATOR" \
            --contract "$BUCKET_CARDINALITY_CONTRACT" \
            --filter-dir "$PLUTO_FILTER_DIR" \
            --filter-prefix "$FILTER_PREFIX" \
            --master-filter "$BUCKET_MASTER_FILTER"
    fi
}

find_latest_checkpoint() {
    local exp_name="$1"
    local checkpoint=""
    if [ -d outputs ]; then
        checkpoint="$({
            find outputs -type f \
                -path "*/outputs/${exp_name}/checkpoints/last.ckpt" \
                -printf '%T@ %p\n' 2>/dev/null || true
        } | sort -nr | head -n 1 | cut -d' ' -f2-)"
    fi
    if [ -z "$checkpoint" ]; then
        echo "Could not find experiment-local final checkpoint for: $exp_name" >&2
        exit 1
    fi
    case "$checkpoint" in /*) ;; *) checkpoint="$(pwd)/${checkpoint}" ;; esac
    echo "$checkpoint"
}

find_latest_checkpoint_optional() {
    local exp_name="$1"
    local checkpoint=""
    if [ -d outputs ]; then
        checkpoint="$({
            find outputs -type f \
                -path "*/outputs/${exp_name}/checkpoints/last.ckpt" \
                -printf '%T@ %p\n' 2>/dev/null || true
        } | sort -nr | head -n 1 | cut -d' ' -f2-)"
    fi
    if [ -n "$checkpoint" ]; then
        case "$checkpoint" in /*) ;; *) checkpoint="$(pwd)/${checkpoint}" ;; esac
    fi
    echo "$checkpoint"
}

prepare_fresh_start() {
    if ! is_enabled "$FRESH_START"; then
        return 0
    fi
    if [ "$START_PHASE" != "auto" ] && [ "$START_PHASE" != "a" ]; then
        echo "Error: FRESH_START=true requires START_PHASE=auto or a." >&2
        exit 1
    fi

    local checkpoint_exp="${CURRICULUM_BASE_EXP}_${PHASE_C_NAME}"
    if [ "$CFG_METHOD_MODE" != "uniform" ]; then
        local phase_key phase_name phase_checkpoint
        for phase_key in a b c; do
            case "$phase_key" in
                a) phase_name="$PHASE_A_NAME" ;;
                b) phase_name="$PHASE_B_NAME" ;;
                c) phase_name="$PHASE_C_NAME" ;;
            esac
            checkpoint_exp="${CURRICULUM_BASE_EXP}_phase${phase_key^^}_${phase_name}"
            phase_checkpoint="$(find_latest_checkpoint_optional "$checkpoint_exp")"
            if [ -n "$phase_checkpoint" ]; then
                echo "Error: FRESH_START=true refuses to archive state while a phase checkpoint exists:" >&2
                echo "  $phase_checkpoint" >&2
                exit 1
            fi
        done
    elif [ -n "$(find_latest_checkpoint_optional "$checkpoint_exp")" ]; then
        echo "Error: FRESH_START=true refuses to archive state while a Uniform checkpoint exists." >&2
        exit 1
    fi

    local timestamp path backup_path
    timestamp="$(date +%Y%m%d_%H%M%S)_$$"
    local state_paths=(
        "$PROTOCOL_SNAPSHOT_PATH"
        "${REPO_ROOT}/artifacts/curriculum_sampling/${CURRICULUM_BASE_EXP}_cumulative_exposure.json"
        "${REPO_ROOT}/artifacts/curriculum_sampling/${CURRICULUM_BASE_EXP}_cumulative_exposure.json.lock"
        "${REPO_ROOT}/artifacts/curriculum_sampling/${CURRICULUM_BASE_EXP}_type_metadata.csv"
    )
    while IFS= read -r path; do
        [ -n "$path" ] && state_paths+=("$path")
    done < <(
        find "${REPO_ROOT}/artifacts/curriculum_sampling" -maxdepth 1 -type f \
            -name "${CURRICULUM_BASE_EXP}_phase*.json*" -print 2>/dev/null || true
    )

    for path in "${state_paths[@]}"; do
        [ -f "$path" ] || continue
        backup_path="${path}.fresh_start_${timestamp}.bak"
        mv -- "$path" "$backup_path"
        echo "Fresh-start backup: $path -> $backup_path"
    done
}

run_lora_train() {
    local experiment_name="$1"
    local checkpoint_arg="$2"
    local checkpoint_path="$3"
    local scenario_filter="$4"
    local epochs="$5"
    local reset_optimizer_moments="$6"
    local require_protocol_match="$7"
    shift 7

    local execution_mode=staged_curriculum
    if [ "$CFG_METHOD_MODE" = "uniform" ]; then
        execution_mode=continuous_uniform
    fi

    local cache_overrides=()
    if [ -n "$FEATURE_CACHE_PATH" ]; then
        cache_overrides+=("cache.cache_path=$FEATURE_CACHE_PATH")
    fi
    local validation_overrides=()
    if [ -n "$VALIDATION_FILTER" ]; then
        validation_overrides+=("+curriculum.validation_filter=$VALIDATION_FILTER")
    fi

    local cmd=(
        "$PYTHON_BIN" scripts/training/finetune_pluto.py
        --config-name training/train_pluto_lora \
        "experiment=$experiment_name"
        "$checkpoint_arg=$checkpoint_path"
        "scenario_filter=$scenario_filter"
        "lora.enabled=$LORA_ENABLED"
        "lora.rank=$LORA_RANK"
        "lora.alpha=$LORA_ALPHA"
        "lora.dropout=$LORA_DROPOUT"
        "lora.lora_lr=$LORA_LR"
        "lora.policy_head_lr=$HEAD_LR"
        "lora.ultra_minimal=$ULTRA_MINIMAL"
        "lora.scheduler_type=$SCHEDULER_TYPE"
        "lora.scheduler_horizon_epochs=$EPOCHS_PHASE_C"
        "lora.scheduler_total_steps=$SCHEDULER_TOTAL_STEPS"
        "lora.scheduler_min_lr=$SCHEDULER_MIN_LR"
        "lora.reset_optimizer_moments_on_resume=$reset_optimizer_moments"
        "lora.training_protocol_id=$PROTOCOL_ID"
        "lora.training_protocol_sha256=$PROTOCOL_SHA256"
        "lora.curriculum_method_id=$METHOD"
        "lora.curriculum_method_sha256=$METHOD_SHA256"
        "lora.curriculum_artifact_bundle_id=$CURRICULUM_ARTIFACT_BUNDLE_ID"
        "lora.curriculum_artifact_bundle_manifest_sha256=$CURRICULUM_ARTIFACT_BUNDLE_MANIFEST_SHA256"
        "lora.execution_mode=$execution_mode"
        "lora.require_protocol_match_on_resume=$require_protocol_match"
        "lora.preserve_l2sp_anchor_on_resume=$PRESERVE_L2SP_ANCHOR_ON_RESUME"
        "lora.preserve_rng_state_on_resume=$PRESERVE_RNG_STATE_ON_RESUME"
        "lr=$LR"
        "weight_decay=$WEIGHT_DECAY"
        "epochs=$epochs"
        "warmup_steps=$WARMUP_STEPS"
        "gradient_clip_val=$GRADIENT_CLIP_VAL"
        "skip_nan_steps=$SKIP_NAN_STEPS"
        "remove_invalid_goals=$REMOVE_INVALID_GOALS"
        "data_loader.params.batch_size=$BATCH_SIZE"
        "+lightning.trainer.params.accumulate_grad_batches=$ACCUMULATE_GRAD_BATCHES"
        "lightning.trainer.params.num_sanity_val_steps=$NUM_SANITY_VAL_STEPS"
        "wandb.name=$experiment_name"
        "seed=$TRAINING_SEED"
        "${cache_overrides[@]}"
        "${validation_overrides[@]}"
        "$@"
    )

    if is_enabled "$DRY_RUN"; then
        printf 'DRY_RUN:'
        printf ' %q' "${cmd[@]}"
        printf '\n'
        return 0
    fi

    "${cmd[@]}"
}

run_phase() {
    local phase_key="$1"
    local phase_name="$2"
    local experiment_name="$3"
    local checkpoint_arg="$4"
    local checkpoint_path="$5"
    local epochs="$6"
    local proportions="$7"
    local phase_start_epoch="$8"
    local pacing_schedule="$9"
    local reset_optimizer_moments=false
    local require_protocol_match=false

    if [ "$phase_key" = "b" ]; then
        reset_optimizer_moments="$RESET_OPTIMIZER_MOMENTS_AT_PHASE_B"
    fi
    if [ "$checkpoint_arg" = "checkpoint" ]; then
        require_protocol_match="$CFG_REQUIRE_PROTOCOL_MATCH_ON_RESUME"
    fi

    if ! is_enabled "$DRY_RUN"; then
        verify_type_routing_metadata_snapshot
    fi
    echo "Phase $phase_key: $phase_name, reset Adam moments=$reset_optimizer_moments"
    if [ "$CFG_METHOD_MODE" = "uniform" ]; then
        run_lora_train \
            "$experiment_name" "$checkpoint_arg" "$checkpoint_path" \
            "$SCENARIO_FILTER_UNIFORM" "$epochs" \
            "$reset_optimizer_moments" "$require_protocol_match"
        return 0
    fi

    local stage_role=all_consolidation
    [ "$phase_key" = "a" ] && stage_role=easy_oriented
    [ "$phase_key" = "b" ] && stage_role=hard_oriented
    run_lora_train \
        "$experiment_name" "$checkpoint_arg" "$checkpoint_path" \
        "$SCENARIO_FILTER_EASY" "$epochs" \
        "$reset_optimizer_moments" "$require_protocol_match" \
        "+curriculum.splits=$CURRICULUM_SPLITS" \
        "+curriculum.sampling_weights=$proportions" \
        "curriculum.pacing_schedule=$pacing_schedule" \
        "curriculum.score_method=$CFG_SCORE_METHOD" \
        "curriculum.bucket_split_rule=quantile_33_33_33" \
        "curriculum.bucketization_mode=$BUCKETIZATION_MODE" \
        "curriculum.percentile_split_seed=$PERCENTILE_SPLIT_SEED" \
        "curriculum.tie_break_mode=$TIE_BREAK_MODE" \
        "curriculum.sampler_mode=$SAMPLER_MODE" \
        "curriculum.phase_name=$phase_name" \
        "+curriculum.phase_start_epoch=$phase_start_epoch" \
        "curriculum.max_repeat_per_scenario=$MAX_REPEAT_PER_SCENARIO" \
        "curriculum.method=$CURRICULUM_METHOD" \
        "curriculum.demonstration_type_mode=$DEMONSTRATION_TYPE_MODE" \
        "curriculum.demonstration_type_metadata_path=$DEMONSTRATION_TYPE_METADATA_PATH" \
        "curriculum.demonstration_type_policy.stage_role=$stage_role" \
        "curriculum.demonstration_type_policy.multiplier_strength=$TYPE_ROUTING_STRENGTH" \
        "curriculum.max_repeat_per_near_duplicate_group=$MAX_REPEAT_PER_NEAR_DUPLICATE_GROUP" \
        "curriculum.near_duplicate_group_weighting=$NEAR_DUPLICATE_GROUP_WEIGHTING" \
        "curriculum.cumulative_exposure_state_path=$CUMULATIVE_EXPOSURE_STATE_PATH" \
        "curriculum.max_cumulative_exposure_per_scenario=$MAX_CUMULATIVE_EXPOSURE_PER_SCENARIO" \
        "curriculum.max_cumulative_exposure_per_near_duplicate_group=$MAX_CUMULATIVE_EXPOSURE_PER_NEAR_DUPLICATE_GROUP" \
        "curriculum.hard_subtype_balance=$HARD_SUBTYPE_BALANCE" \
        "curriculum.random_seed=$SAMPLER_SEED" \
        "curriculum.sampling_log_path=artifacts/curriculum_sampling/${experiment_name}.json" \
        "curriculum.filter_file_path=${PLUTO_FILTER_DIR}/${SCENARIO_FILTER_EASY}.yaml"
}

main() {
    cd "$REPO_ROOT"
    case "$START_PHASE" in
        auto|a|b|c) ;;
        *)
            echo "Error: START_PHASE must be one of: auto, a, b, c." >&2
            exit 1
            ;;
    esac
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/scripts/env_bootstrap.sh"
    ensure_method_filters
    if ! is_enabled "$DRY_RUN"; then
        prepare_fresh_start
    fi
    if ! is_enabled "$DRY_RUN"; then
        freeze_type_routing_metadata
    fi
    if ! is_enabled "$DRY_RUN"; then
        "$PYTHON_BIN" "$CONFIG_RESOLVER" \
            --protocol "$TRAINING_PROTOCOL_CONFIG" \
            --method "$METHOD_CONFIG" \
            --format json \
            --output "$PROTOCOL_SNAPSHOT_PATH"
    fi

    echo "============================================================"
    echo "PLUTO LoRA: method=$METHOD, artifact=$CURRICULUM_VERSION"
    echo "Protocol: $PROTOCOL_ID ($PROTOCOL_SHA256)"
    echo "Method config: $CFG_METHOD_PATH ($METHOD_SHA256)"
    echo "Phase policy: $PHASE_POLICY"
    echo "Resume continuity: L2-SP anchor=$PRESERVE_L2SP_ANCHOR_ON_RESUME, RNG=$PRESERVE_RNG_STATE_ON_RESUME"
    echo "Scheduler: $SCHEDULER_TYPE, steps=$SCHEDULER_TOTAL_STEPS, warmup=$WARMUP_STEPS, min LR=$SCHEDULER_MIN_LR, LoRA LR=$LORA_LR, head LR=$HEAD_LR"
    if [ "$CFG_METHOD_MODE" = "uniform" ]; then
        echo "Execution: continuous Uniform FT for $EPOCHS_PHASE_C epochs"
        echo "Phase transitions: disabled; optimizer reset: disabled"
    else
        echo "Cumulative epochs: $EPOCHS_PHASE_A/$EPOCHS_PHASE_B/$EPOCHS_PHASE_C"
        echo "A->B Adam reset: $RESET_OPTIMIZER_MOMENTS_AT_PHASE_B; B->C reset: false"
    fi
    echo "Seed: training=$TRAINING_SEED sampler=$SAMPLER_SEED"
    if [ -n "$FEATURE_CACHE_PATH" ]; then
        echo "Feature cache: $FEATURE_CACHE_PATH"
    else
        echo "Feature cache: disabled"
    fi
    echo "Experiment base: $CURRICULUM_BASE_EXP"
    echo "Resolved snapshot: $PROTOCOL_SNAPSHOT_PATH"
    echo "Dry run: $DRY_RUN"
    echo "Start phase: $START_PHASE"
    echo "Fresh start: $FRESH_START"
    echo "============================================================"

    if [ ! -f "$PRETRAINED_CKPT" ]; then
        echo "Error: pretrained checkpoint not found: $PRETRAINED_CKPT" >&2
        exit 1
    fi

    local phase_a_exp phase_b_exp phase_c_exp
    if [ "$CFG_METHOD_MODE" = "uniform" ]; then
        if [ "$START_PHASE" != "auto" ] && [ "$START_PHASE" != "a" ]; then
            echo "Error: START_PHASE is only valid for staged curriculum methods." >&2
            exit 1
        fi
        phase_c_exp="${CURRICULUM_BASE_EXP}_${PHASE_C_NAME}"
        echo "Uniform FT is a single continuous run; A/B curriculum phases are skipped."
        run_lora_train \
            "$phase_c_exp" pretrained_ckpt "$PRETRAINED_CKPT" \
            "$SCENARIO_FILTER_UNIFORM" "$EPOCHS_PHASE_C" \
            false false
        local uniform_final_ckpt
        if is_enabled "$DRY_RUN"; then
            uniform_final_ckpt="<${phase_c_exp}/checkpoints/last.ckpt>"
        else
            uniform_final_ckpt="$(find_latest_checkpoint "$phase_c_exp")"
        fi
        echo "Completed $METHOD experiment: $uniform_final_ckpt"
        return 0
    fi

    phase_a_exp="${CURRICULUM_BASE_EXP}_phaseA_${PHASE_A_NAME}"
    phase_b_exp="${CURRICULUM_BASE_EXP}_phaseB_${PHASE_B_NAME}"
    phase_c_exp="${CURRICULUM_BASE_EXP}_phaseC_${PHASE_C_NAME}"

    local effective_start_phase="$START_PHASE"
    if [ "$effective_start_phase" = "auto" ]; then
        local existing_phase_a_ckpt existing_phase_b_ckpt existing_phase_c_ckpt
        existing_phase_c_ckpt="$(find_latest_checkpoint_optional "$phase_c_exp")"
        existing_phase_b_ckpt="$(find_latest_checkpoint_optional "$phase_b_exp")"
        existing_phase_a_ckpt="$(find_latest_checkpoint_optional "$phase_a_exp")"
        if [ -n "$existing_phase_c_ckpt" ]; then
            echo "Auto-resume: completed Phase C checkpoint already exists: $existing_phase_c_ckpt"
            echo "Completed $METHOD experiment: $existing_phase_c_ckpt"
            return 0
        elif [ -n "$existing_phase_b_ckpt" ]; then
            effective_start_phase=c
        elif [ -n "$existing_phase_a_ckpt" ]; then
            effective_start_phase=b
        else
            effective_start_phase=a
        fi
        echo "Auto-resume selected start phase: $effective_start_phase"
    fi

    local phase_a_ckpt
    if [ "$effective_start_phase" = "a" ]; then
        run_phase a "$PHASE_A_NAME" "$phase_a_exp" pretrained_ckpt "$PRETRAINED_CKPT" \
            "$EPOCHS_PHASE_A" "$PHASE_A_TARGET_PROPORTIONS" 0 "$PHASE_A_PACING_SCHEDULE"
        if is_enabled "$DRY_RUN"; then
            phase_a_ckpt="<${phase_a_exp}/checkpoints/last.ckpt>"
        else
            phase_a_ckpt="$(find_latest_checkpoint "$phase_a_exp")"
        fi
    else
        phase_a_ckpt="$(find_latest_checkpoint "$phase_a_exp")"
        echo "Skipping Phase A; resuming from: $phase_a_ckpt"
    fi

    local phase_b_ckpt
    if [ "$effective_start_phase" = "a" ] || [ "$effective_start_phase" = "b" ]; then
        run_phase b "$PHASE_B_NAME" "$phase_b_exp" checkpoint "$phase_a_ckpt" \
            "$EPOCHS_PHASE_B" "$PHASE_B_TARGET_PROPORTIONS" "$EPOCHS_PHASE_A" \
            "$PHASE_B_PACING_SCHEDULE"
        if is_enabled "$DRY_RUN"; then
            phase_b_ckpt="<${phase_b_exp}/checkpoints/last.ckpt>"
        else
            phase_b_ckpt="$(find_latest_checkpoint "$phase_b_exp")"
        fi
    else
        phase_b_ckpt="$(find_latest_checkpoint "$phase_b_exp")"
        echo "Skipping Phases A/B; resuming from: $phase_b_ckpt"
    fi

    run_phase c "$PHASE_C_NAME" "$phase_c_exp" checkpoint "$phase_b_ckpt" \
        "$EPOCHS_PHASE_C" "$PHASE_C_TARGET_PROPORTIONS" "$EPOCHS_PHASE_B" \
        "$PHASE_C_PACING_SCHEDULE"
    local final_ckpt
    if is_enabled "$DRY_RUN"; then
        final_ckpt="<${phase_c_exp}/checkpoints/last.ckpt>"
    else
        final_ckpt="$(find_latest_checkpoint "$phase_c_exp")"
    fi
    echo "Completed $METHOD experiment: $final_ckpt"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
main "$@"
fi
