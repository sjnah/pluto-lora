#!/bin/bash
# PLUTO LoRA fine-tuning experiment using the uniform-principle curriculum.
#
# This is the uniform baseline for curriculum comparison. It uses the shared
# common-valid all-scenario filter instead of LLM/rule/loss-ranked difficulty
# stages, and writes to curriculum_lora_uniform for clear method naming.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
NUPLAN_DEVKIT_ROOT="${WORKSPACE_ROOT}/nuplan-devkit"
cd "$REPO_ROOT"

# ============================================================================
# EASY PARAMETER TUNING
# ============================================================================

PRETRAINED_CKPT="${REPO_ROOT}/checkpoints/pluto_1M_aux_cil.ckpt"

LORA_ENABLED=true
LORA_RANK=8
LORA_ALPHA=16.0
LORA_DROPOUT=0.05
LORA_LR=2e-4
HEAD_LR=6e-6

LR=1e-5
WEIGHT_DECAY=0.05
EPOCHS=14
WARMUP_STEPS=200

GRADIENT_CLIP_VAL=0.5
SKIP_NAN_STEPS=true
REMOVE_INVALID_GOALS=false
ULTRA_MINIMAL=true

BATCH_SIZE=4
ACCUMULATE_GRAD_BATCHES=8

SCENARIO_FILTER_UNIFORM="uniform_train_all"
EXPERIMENT="curriculum_lora_uniform"

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

# Set up Python/runtime paths. Supports conda, .venv, or an already-active env.
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/env_bootstrap.sh"

echo "=============================================="
echo "PLUTO LoRA Curriculum Experiment (Uniform)"
echo "=============================================="
echo "Scenario filter: $SCENARIO_FILTER_UNIFORM"
echo "Experiment: $EXPERIMENT"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE, accumulation: $ACCUMULATE_GRAD_BATCHES"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

if [ ! -f "$PRETRAINED_CKPT" ]; then
    echo "Error: Pretrained checkpoint not found: $PRETRAINED_CKPT"
    exit 1
fi

echo ""
echo "=============================================="
echo "EXPERIMENT: Uniform-principle curriculum fine-tuning"
echo "=============================================="

if [ "$LORA_ENABLED" = "true" ]; then
    python scripts/training/finetune_pluto.py \
        --config-name training/train_pluto_lora \
        experiment="$EXPERIMENT" \
        pretrained_ckpt="$PRETRAINED_CKPT" \
        scenario_filter="$SCENARIO_FILTER_UNIFORM" \
        lora.enabled=true \
        lora.rank="$LORA_RANK" \
        lora.alpha="$LORA_ALPHA" \
        lora.dropout="$LORA_DROPOUT" \
        lora.lora_lr="$LORA_LR" \
        lora.policy_head_lr="$HEAD_LR" \
        lora.ultra_minimal="$ULTRA_MINIMAL" \
        lr="$LR" \
        weight_decay="$WEIGHT_DECAY" \
        epochs="$EPOCHS" \
        warmup_steps="$WARMUP_STEPS" \
        gradient_clip_val="$GRADIENT_CLIP_VAL" \
        skip_nan_steps="$SKIP_NAN_STEPS" \
        remove_invalid_goals="$REMOVE_INVALID_GOALS" \
        data_loader.params.batch_size="$BATCH_SIZE" \
        +trainer.params.accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES" \
        wandb.name="$EXPERIMENT"
else
    python scripts/training/finetune_pluto.py \
        --config-name training/train_pluto_head_only \
        experiment="$EXPERIMENT" \
        pretrained_ckpt="$PRETRAINED_CKPT" \
        scenario_filter="$SCENARIO_FILTER_UNIFORM" \
        lora.enabled=false \
        lora.ultra_minimal="$ULTRA_MINIMAL" \
        lora.policy_head_lr="$HEAD_LR" \
        lr="$LR" \
        weight_decay="$WEIGHT_DECAY" \
        epochs="$EPOCHS" \
        warmup_steps="$WARMUP_STEPS" \
        data_loader.params.batch_size="$BATCH_SIZE" \
        +trainer.params.accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES" \
        wandb.name="$EXPERIMENT"
fi

UNIFORM_CKPT="$(find_latest_checkpoint "$EXPERIMENT")"

echo ""
echo "=============================================="
echo "Done: Uniform-principle curriculum experiment complete"
echo "=============================================="
echo "Uniform curriculum checkpoint: $UNIFORM_CKPT"
echo "Use run_lora_experiment_llmbased.sh for the LLM-based curriculum experiment."
