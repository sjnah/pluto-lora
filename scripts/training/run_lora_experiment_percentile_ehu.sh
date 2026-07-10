#!/bin/bash
# Shared PLUTO LoRA percentile-tercile curriculum runner.
#
# Design: easy warm-up -> mild hard focus -> uniform consolidation. This is not
# a direct Rampp reproduction; it combines easy-to-hard curriculum, mild
# hard-example emphasis, and final uniform consolidation.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
NUPLAN_DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-${WORKSPACE_ROOT}/nuplan-devkit}"
PLUTO_FILTER_DIR="${REPO_ROOT}/config/scenario_filter"
cd "$REPO_ROOT"

METHOD="${METHOD:-llm}" # llm, rule, loss, mpoc
METHOD_LABEL="${METHOD_LABEL:-${METHOD}}"
CURRICULUM_VERSION="${CURRICULUM_VERSION:-v4.3.13}"
PERCENTILE_SPLIT_SEED="${PERCENTILE_SPLIT_SEED:-42}"
SAMPLER_SEED="${SAMPLER_SEED:-${PERCENTILE_SPLIT_SEED}}"

PRETRAINED_CKPT="${REPO_ROOT}/checkpoints/pluto_1M_aux_cil.ckpt"

LORA_ENABLED=true
LORA_RANK=4
LORA_ALPHA=8.0
LORA_DROPOUT=0.05
LORA_LR=5e-5
HEAD_LR1=1e-5 # 0.0
HEAD_LR2=1e-5
HEAD_LR3=3e-5

LR=1e-5
WEIGHT_DECAY=0.05
# The previous common curriculum shape used 12 cumulative epochs. Integer
# epochs cannot represent 20/40/40 exactly, so the default keeps the same total
# epoch/update budget with the closest 2/5/5 split.
EPOCHS_PHASE_A="${EPOCHS_PHASE_A:-2}"
EPOCHS_PHASE_B="${EPOCHS_PHASE_B:-7}"
EPOCHS_PHASE_C="${EPOCHS_PHASE_C:-12}"
WARMUP_STEPS=100

GRADIENT_CLIP_VAL=0.5
SKIP_NAN_STEPS=true
REMOVE_INVALID_GOALS=false
ULTRA_MINIMAL=true

BATCH_SIZE=4
ACCUMULATE_GRAD_BATCHES=8

FILTER_PREFIX="${FILTER_PREFIX:-${METHOD}_percentile_ehu}"
SCENARIO_FILTER_EASY="${FILTER_PREFIX}_train_easy"
SCENARIO_FILTER_MEDIUM="${FILTER_PREFIX}_train_medium"
SCENARIO_FILTER_HARD="${FILTER_PREFIX}_train_hard"
CURRICULUM_SPLITS="[$SCENARIO_FILTER_EASY,$SCENARIO_FILTER_MEDIUM,$SCENARIO_FILTER_HARD]"

PHASE_A_NAME="easy_warmup"
PHASE_B_NAME="mild_hard_focus"
PHASE_C_NAME="uniform_consolidation"
PHASE_A_TARGET_PROPORTIONS="${PHASE_A_TARGET_PROPORTIONS:-[0.50,0.40,0.10]}"
PHASE_B_TARGET_PROPORTIONS="${PHASE_B_TARGET_PROPORTIONS:-[0.267,0.333,0.400]}"
PHASE_C_TARGET_PROPORTIONS="${PHASE_C_TARGET_PROPORTIONS:-[0.333333,0.333333,0.333334]}"
MAX_REPEAT_PER_SCENARIO="${MAX_REPEAT_PER_SCENARIO:-4}"
HARD_SUBTYPE_BALANCE="${HARD_SUBTYPE_BALANCE:-false}"

BUCKETIZATION_MODE="percentile_tercile"
TIE_BREAK_MODE="${TIE_BREAK_MODE:-stable_hash}"
SAMPLER_MODE="${SAMPLER_MODE:-exact_bucket_quota}"

CURRICULUM_BASE_EXP="${CURRICULUM_BASE_EXP:-curriculum_lora_${METHOD}_percentile_ehu_${CURRICULUM_VERSION}}"

ensure_percentile_filters() {
    local missing=0
    for filter_name in "$SCENARIO_FILTER_EASY" "$SCENARIO_FILTER_MEDIUM" "$SCENARIO_FILTER_HARD"; do
        local target="${PLUTO_FILTER_DIR}/${filter_name}.yaml"
        if [ ! -f "$target" ]; then
            echo "Missing PLUTO percentile scenario filter: $target"
            missing=1
        fi
    done
    if [ "$missing" -ne 0 ]; then
        echo ""
        echo "Generate percentile-tercile filters first, for example:"
        echo "  cd ${WORKSPACE_ROOT}/llm-taxonomy"
        echo "  python scripts/experiments/pluto/create_percentile_tercile_filters.py \\"
        echo "    --methods ${METHOD} \\"
        echo "    --seed ${PERCENTILE_SPLIT_SEED} \\"
        echo "    --copy-to-pluto-config"
        exit 1
    fi
}

find_latest_checkpoint() {
    local exp_name="$1"
    local checkpoint=""

    if [ -d outputs ]; then
        for exp_dir in $(find outputs -type d -name "$exp_name" 2>/dev/null | sort -r); do
            local parent_dir
            parent_dir="$(dirname "$(dirname "$exp_dir")")"
            local candidate="${parent_dir}/checkpoints/last.ckpt"
            if [ -f "$candidate" ]; then
                checkpoint="$candidate"
                break
            fi
        done
    fi

    if [ -z "$checkpoint" ]; then
        echo "Could not find checkpoint for experiment: $exp_name"
        exit 1
    fi

    if [ "${checkpoint:0:1}" != "/" ]; then
        checkpoint="$(pwd)/${checkpoint}"
    fi
    echo "$checkpoint"
}

run_lora_train() {
    local experiment_name="$1"
    local checkpoint_arg="$2"
    local checkpoint_path="$3"
    local scenario_filter="$4"
    local epochs="$5"
    local head_lr="$6"
    shift 6

    if [ "$LORA_ENABLED" = "true" ]; then
        python scripts/training/finetune_pluto.py \
            --config-name training/train_pluto_lora \
            experiment="$experiment_name" \
            "$checkpoint_arg=$checkpoint_path" \
            scenario_filter="$scenario_filter" \
            lora.enabled=true \
            lora.rank="$LORA_RANK" \
            lora.alpha="$LORA_ALPHA" \
            lora.dropout="$LORA_DROPOUT" \
            lora.lora_lr="$LORA_LR" \
            lora.policy_head_lr="$head_lr" \
            lora.ultra_minimal="$ULTRA_MINIMAL" \
            lr="$LR" \
            weight_decay="$WEIGHT_DECAY" \
            epochs="$epochs" \
            warmup_steps="$WARMUP_STEPS" \
            gradient_clip_val="$GRADIENT_CLIP_VAL" \
            skip_nan_steps="$SKIP_NAN_STEPS" \
            remove_invalid_goals="$REMOVE_INVALID_GOALS" \
            data_loader.params.batch_size="$BATCH_SIZE" \
            +lightning.trainer.params.accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES" \
            lightning.trainer.params.num_sanity_val_steps=0 \
            wandb.name="$experiment_name" \
            "$@"
    else
        python scripts/training/finetune_pluto.py \
            --config-name training/train_pluto_head_only \
            experiment="$experiment_name" \
            "$checkpoint_arg=$checkpoint_path" \
            scenario_filter="$scenario_filter" \
            lora.enabled=false \
            lora.ultra_minimal="$ULTRA_MINIMAL" \
            lora.policy_head_lr="$head_lr" \
            lr="$LR" \
            weight_decay="$WEIGHT_DECAY" \
            epochs="$epochs" \
            warmup_steps="$WARMUP_STEPS" \
            data_loader.params.batch_size="$BATCH_SIZE" \
            +lightning.trainer.params.accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES" \
            lightning.trainer.params.num_sanity_val_steps=0 \
            wandb.name="$experiment_name" \
            "$@"
    fi
}

run_curriculum_phase() {
    local phase_name="$1"
    local experiment_name="$2"
    local checkpoint_arg="$3"
    local checkpoint_path="$4"
    local epochs="$5"
    local head_lr="$6"
    local proportions="$7"

    echo "Phase: ${phase_name}, target proportions [easy,medium,hard]: ${proportions}"
    run_lora_train \
        "$experiment_name" \
        "$checkpoint_arg" \
        "$checkpoint_path" \
        "$SCENARIO_FILTER_EASY" \
        "$epochs" \
        "$head_lr" \
        "+curriculum.splits=$CURRICULUM_SPLITS" \
        "+curriculum.sampling_weights=$proportions" \
        "curriculum.score_method=$METHOD" \
        "curriculum.bucket_split_rule=quantile_33_33_33" \
        "curriculum.bucketization_mode=$BUCKETIZATION_MODE" \
        "curriculum.percentile_split_seed=$PERCENTILE_SPLIT_SEED" \
        "curriculum.tie_break_mode=$TIE_BREAK_MODE" \
        "curriculum.sampler_mode=$SAMPLER_MODE" \
        "curriculum.phase_name=$phase_name" \
        "curriculum.max_repeat_per_scenario=$MAX_REPEAT_PER_SCENARIO" \
        "curriculum.hard_subtype_balance=$HARD_SUBTYPE_BALANCE" \
        "curriculum.random_seed=$SAMPLER_SEED" \
        "curriculum.sampling_log_path=artifacts/curriculum_sampling/${experiment_name}.json" \
        "curriculum.filter_file_path=${PLUTO_FILTER_DIR}/${SCENARIO_FILTER_EASY}.yaml"
}

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/env_bootstrap.sh"

ensure_percentile_filters

echo "=============================================="
echo "PLUTO LoRA Percentile-EHU Curriculum (${METHOD_LABEL}, ${CURRICULUM_VERSION})"
echo "=============================================="
echo "Filters: $SCENARIO_FILTER_EASY, $SCENARIO_FILTER_MEDIUM, $SCENARIO_FILTER_HARD"
echo "Bucketization: $BUCKETIZATION_MODE, tie break: $TIE_BREAK_MODE, split seed: $PERCENTILE_SPLIT_SEED"
echo "Sampler: $SAMPLER_MODE, sampler seed: $SAMPLER_SEED, max repeat: $MAX_REPEAT_PER_SCENARIO"
echo "Experiment base: $CURRICULUM_BASE_EXP"
echo "Epochs cumulative: $EPOCHS_PHASE_A/$EPOCHS_PHASE_B/$EPOCHS_PHASE_C"
echo "Batch size: $BATCH_SIZE, accumulation: $ACCUMULATE_GRAD_BATCHES"
echo ""

if [ ! -f "$PRETRAINED_CKPT" ]; then
    echo "Error: Pretrained checkpoint not found: $PRETRAINED_CKPT"
    exit 1
fi

PHASE_A_EXP="${CURRICULUM_BASE_EXP}_phaseA_${PHASE_A_NAME}"
run_curriculum_phase "$PHASE_A_NAME" "$PHASE_A_EXP" "pretrained_ckpt" "$PRETRAINED_CKPT" "$EPOCHS_PHASE_A" "$HEAD_LR1" "$PHASE_A_TARGET_PROPORTIONS"
PHASE_A_CKPT="$(find_latest_checkpoint "$PHASE_A_EXP")"
echo "Phase A checkpoint: $PHASE_A_CKPT"

PHASE_B_EXP="${CURRICULUM_BASE_EXP}_phaseB_${PHASE_B_NAME}"
run_curriculum_phase "$PHASE_B_NAME" "$PHASE_B_EXP" "checkpoint" "$PHASE_A_CKPT" "$EPOCHS_PHASE_B" "$HEAD_LR2" "$PHASE_B_TARGET_PROPORTIONS"
PHASE_B_CKPT="$(find_latest_checkpoint "$PHASE_B_EXP")"
echo "Phase B checkpoint: $PHASE_B_CKPT"

PHASE_C_EXP="${CURRICULUM_BASE_EXP}_phaseC_${PHASE_C_NAME}"
run_curriculum_phase "$PHASE_C_NAME" "$PHASE_C_EXP" "checkpoint" "$PHASE_B_CKPT" "$EPOCHS_PHASE_C" "$HEAD_LR3" "$PHASE_C_TARGET_PROPORTIONS"
CURRICULUM_CKPT="$(find_latest_checkpoint "$PHASE_C_EXP")"

echo ""
echo "=============================================="
echo "Done: Percentile-EHU curriculum experiment complete"
echo "=============================================="
echo "Curriculum checkpoint: $CURRICULUM_CKPT"
