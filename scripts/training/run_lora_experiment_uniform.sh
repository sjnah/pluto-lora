#!/bin/bash
# PLUTO LoRA fine-tuning experiment using the uniform fine-tuning baseline.
#
# This baseline keeps the same 3-stage experiment shape as curriculum methods,
# but every stage uses the shared common-valid all-scenario filter instead of
# difficulty buckets. Stage 2/3 resume checkpoint state instead of resetting the
# Lightning loop, so max_epochs values are cumulative.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
NUPLAN_DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-${WORKSPACE_ROOT}/nuplan-devkit}"
cd "$REPO_ROOT"

# ============================================================================
# EASY PARAMETER TUNING
# ============================================================================

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
EPOCHS_STAGE1=4
EPOCHS_STAGE2=8
EPOCHS_STAGE3=12
WARMUP_STEPS=100

GRADIENT_CLIP_VAL=0.5
SKIP_NAN_STEPS=true
REMOVE_INVALID_GOALS=false
ULTRA_MINIMAL=true

BATCH_SIZE=4
ACCUMULATE_GRAD_BATCHES=8

SCENARIO_FILTER_UNIFORM="uniform_train_all"
UNIFORM_CURRICULUM_VERSION="${UNIFORM_CURRICULUM_VERSION:-v2.3.9}"
CURRICULUM_BASE_EXP="${CURRICULUM_BASE_EXP:-curriculum_lora_uniform_${UNIFORM_CURRICULUM_VERSION}}"

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

# Set up Python/runtime paths. Supports conda, .venv, or an already-active env.
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/env_bootstrap.sh"

echo "=============================================="
echo "PLUTO LoRA Uniform Fine-Tuning Baseline (${UNIFORM_CURRICULUM_VERSION})"
echo "=============================================="
echo "Scenario filter: $SCENARIO_FILTER_UNIFORM"
echo "Experiment base: $CURRICULUM_BASE_EXP"
echo "Max epochs: $EPOCHS_STAGE1/$EPOCHS_STAGE2/$EPOCHS_STAGE3 (effective increments: $EPOCHS_STAGE1/$((EPOCHS_STAGE2 - EPOCHS_STAGE1))/$((EPOCHS_STAGE3 - EPOCHS_STAGE2)))"
echo "Stage resume: keep Lightning optimizer/scheduler/loop state across stages"
echo "LR note: stage 2/3 resume optimizer state, so stage-specific LR args are not independent"
echo "Batch size: $BATCH_SIZE, accumulation: $ACCUMULATE_GRAD_BATCHES"
echo ""
echo

if [ ! -f "$PRETRAINED_CKPT" ]; then
    echo "Error: Pretrained checkpoint not found: $PRETRAINED_CKPT"
    exit 1
fi

echo ""
echo "=============================================="
echo "EXPERIMENT: Uniform fine-tuning baseline (${UNIFORM_CURRICULUM_VERSION})"
echo "=============================================="

STAGE1_EXP="${CURRICULUM_BASE_EXP}_stage1_raw"
echo "Stage 1/3: uniform raw distribution"
run_lora_train \
    "$STAGE1_EXP" \
    "pretrained_ckpt" \
    "$PRETRAINED_CKPT" \
    "$SCENARIO_FILTER_UNIFORM" \
    "$EPOCHS_STAGE1" \
    "$HEAD_LR1"
STAGE1_CKPT="$(find_latest_checkpoint "$STAGE1_EXP")"
echo "Stage 1 checkpoint: $STAGE1_CKPT"

STAGE2_EXP="${CURRICULUM_BASE_EXP}_stage2_uniform"
echo "Stage 2/3: uniform distribution, continuing from stage 1"
run_lora_train \
    "$STAGE2_EXP" \
    "checkpoint" \
    "$STAGE1_CKPT" \
    "$SCENARIO_FILTER_UNIFORM" \
    "$EPOCHS_STAGE2" \
    "$HEAD_LR2"
STAGE2_CKPT="$(find_latest_checkpoint "$STAGE2_EXP")"
echo "Stage 2 checkpoint: $STAGE2_CKPT"

STAGE3_EXP="${CURRICULUM_BASE_EXP}_stage3_uniform"
echo "Stage 3/3: uniform distribution, continuing from stage 2"
run_lora_train \
    "$STAGE3_EXP" \
    "checkpoint" \
    "$STAGE2_CKPT" \
    "$SCENARIO_FILTER_UNIFORM" \
    "$EPOCHS_STAGE3" \
    "$HEAD_LR3"
UNIFORM_CKPT="$(find_latest_checkpoint "$STAGE3_EXP")"

echo ""
echo "=============================================="
echo "Done: Uniform fine-tuning baseline complete"
echo "=============================================="
echo "Uniform checkpoint: $UNIFORM_CKPT"
echo "Use run_lora_experiment_llmbased.sh for the LLM-guided curriculum experiment."
