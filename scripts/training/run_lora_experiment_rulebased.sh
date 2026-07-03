#!/bin/bash
# PLUTO LoRA Fine-tuning Experiment Script (Rule-based Score Version)
# 
# This script runs the complete LoRA fine-tuning comparison using rule-based
# difficulty scores (difficulty_score) for scenario filtering:
# 1. Uniform fine-tuning (all difficulty levels mixed)
# 2. Curriculum fine-tuning (3-stage progressive training)
#
# Features:
# - Encoder-only LoRA (encoder_blocks attention layers)
# - Ultra-minimal head fine-tuning (only mlp.3)
# - Identical total update steps for fair comparison
# - Uses rule-based score percentile splits (rulebased_train_*.yaml)

set -e

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
NUPLAN_DEVKIT_ROOT="${WORKSPACE_ROOT}/nuplan-devkit"
cd "$REPO_ROOT"

# ============================================================================
# EASY PARAMETER TUNING - Modify these values as needed
# ============================================================================

# Pretrained checkpoint
PRETRAINED_CKPT="${REPO_ROOT}/checkpoints/pluto_1M_aux_cil.ckpt"

# LoRA Configuration
LORA_ENABLED=true              # Enable/disable LoRA
LORA_RANK=8                    # LoRA rank (lower = fewer parameters)
LORA_ALPHA=16.0                # LoRA alpha scaling factor
LORA_DROPOUT=0.05              # LoRA dropout probability
LORA_LR=2e-4                   # Learning rate for LoRA parameters
HEAD_LR=1e-5                   # Learning rate for head parameters (mlp.3)

# Training Hyperparameters
LR=1e-5                        # Base learning rate (used if LoRA disabled)
WEIGHT_DECAY=0.05              # Weight decay
EPOCHS=5                       # Epochs for uniform training
EPOCHS_STAGE1=4             # Epochs per curriculum stage 1
EPOCHS_STAGE2=6             # Epochs per curriculum stage 2
EPOCHS_STAGE3=10             # Epochs per curriculum stage 3
WARMUP_STEPS=200               # Warmup steps

# Training Stability
GRADIENT_CLIP_VAL=0.5          # Gradient clipping value
SKIP_NAN_STEPS=true           # Skip steps with NaN/Inf loss
REMOVE_INVALID_GOALS=false     # Use audited common-valid scenario filters directly

# Ultra-minimal mode (only mlp.3 trainable when LoRA disabled)
ULTRA_MINIMAL=true

# ============================================================================
# Batch Size Configuration
# ============================================================================
# VRAM 11GB: Batch 4 추천 (OOM 발생 시 2로 낮추고 Accumulation을 16으로 올리세요)
BATCH_SIZE=4
ACCUMULATE_GRAD_BATCHES=8  # 4 * 8 = 32 (Target Effective Batch)

# Scenario Filters (Rule-based score percentile splits)
# These filters are created by llm-taxonomy/scripts/experiments/pluto/create_rulebased_filters.py
SCENARIO_FILTER_UNIFORM="rulebased_train_all"
SCENARIO_FILTER_STAGE1="rulebased_train_easy"
SCENARIO_FILTER_STAGE2="rulebased_train_medium"
SCENARIO_FILTER_STAGE3="rulebased_train_hard"

# Experiment Names (suffixed with _rulebased to distinguish from LLM-based version)
UNIFORM_EXP="uniform_lora_finetune_rulebased"
CURRICULUM_BASE_EXP="curriculum_lora_rulebased"

# ============================================================================
# Setup
# ============================================================================

# Set up Python/runtime paths. Supports conda, .venv, or an already-active env.
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/env_bootstrap.sh"
echo "PYTHONPATH set to: ${PYTHONPATH}"
echo "NUPLAN_DATA_ROOT set to: ${NUPLAN_DATA_ROOT}"

echo "=============================================="
echo "PLUTO LoRA Fine-tuning Experiment (Rule-based Score)"
echo "=============================================="
echo ""
echo "This will run:"
echo "  1. Uniform fine-tuning (all difficulties mixed with weights [0.5, 0.3, 0.2])"
echo "     - Total epochs: $((EPOCHS_STAGE1 + EPOCHS_STAGE2 + EPOCHS_STAGE3)) (same as curriculum total)"
echo "     - Using rule-based score percentile splits"
echo "  2. Curriculum fine-tuning (3 stages)"
echo "     - Stage 1: Easy scenarios ($EPOCHS_STAGE1 epochs)"
echo "     - Stage 2: Easy + Medium [0.4, 0.4, 0.2] ($EPOCHS_STAGE2 epochs)"
echo "     - Stage 3: Easy + Medium + Hard [0.2, 0.3, 0.5] ($EPOCHS_STAGE3 epochs)"
echo "     - Total epochs: $((EPOCHS_STAGE1 + EPOCHS_STAGE2 + EPOCHS_STAGE3))"
echo "     - Using rule-based score percentile splits"
echo ""
echo "Configuration:"
echo "  LoRA enabled: $LORA_ENABLED"
if [ "$LORA_ENABLED" = "true" ]; then
    echo "  LoRA rank: $LORA_RANK"
    echo "  LoRA alpha: $LORA_ALPHA"
    echo "  LoRA LR: $LORA_LR"
    echo "  Head LR: $HEAD_LR"
else
    echo "  Ultra-minimal mode: $ULTRA_MINIMAL (only mlp.3 trainable)"
    echo "  Learning rate: $LR"
fi
echo "  Weight decay: $WEIGHT_DECAY"
echo "  Epochs per stage: $EPOCHS_STAGE1, $EPOCHS_STAGE2, $EPOCHS_STAGE3"
echo "  Warmup steps: $WARMUP_STEPS"
echo "  Gradient clip: $GRADIENT_CLIP_VAL"
echo "  Skip NaN steps: $SKIP_NAN_STEPS"
echo "  Scenario filters: Rule-based score percentile splits"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Check pretrained checkpoint
if [ ! -f "$PRETRAINED_CKPT" ]; then
    echo "❌ Error: Pretrained checkpoint not found: $PRETRAINED_CKPT"
    echo "Please download or specify the correct path."
    exit 1
fi



# ============================================================================
# Experiment 2: Curriculum Fine-tuning
# ============================================================================

echo ""
echo "=============================================="
echo "EXPERIMENT 2: Curriculum Fine-tuning (Rule-based Score)"
echo "=============================================="
echo ""

# ============================================================================
# Stage 1: Easy scenarios
# ============================================================================

echo ""
echo "Stage 1/3: Training on EASY scenarios (rule-based score percentile)..."
STAGE1_EXP="${CURRICULUM_BASE_EXP}_stage1_low"

if [ "$LORA_ENABLED" = "true" ]; then
    python scripts/training/finetune_pluto.py \
        --config-name training/train_pluto_lora \
        experiment=$STAGE1_EXP \
        pretrained_ckpt=$PRETRAINED_CKPT \
        scenario_filter=$SCENARIO_FILTER_STAGE1 \
        lora.enabled=true \
        lora.rank=$LORA_RANK \
        lora.alpha=$LORA_ALPHA \
        lora.dropout=$LORA_DROPOUT \
        lora.lora_lr=$LORA_LR \
        lora.policy_head_lr=$HEAD_LR \
        lora.ultra_minimal=$ULTRA_MINIMAL \
        lr=$LR \
        weight_decay=$WEIGHT_DECAY \
        epochs=$EPOCHS_STAGE1 \
        warmup_steps=$WARMUP_STEPS \
        data_loader.params.batch_size=$BATCH_SIZE \
        +trainer.params.accumulate_grad_batches=$ACCUMULATE_GRAD_BATCHES \
        wandb.name="${STAGE1_EXP}"
else
    python scripts/training/finetune_pluto.py \
        --config-name training/train_pluto_head_only \
        experiment=$STAGE1_EXP \
        pretrained_ckpt=$PRETRAINED_CKPT \
        scenario_filter=$SCENARIO_FILTER_STAGE1 \
        lora.enabled=false \
        lora.ultra_minimal=$ULTRA_MINIMAL \
        lora.policy_head_lr=$LR \
        lr=$LR \
        weight_decay=$WEIGHT_DECAY \
        epochs=$EPOCHS_STAGE1 \
        warmup_steps=$WARMUP_STEPS \
        data_loader.params.batch_size=$BATCH_SIZE \
        +trainer.params.accumulate_grad_batches=$ACCUMULATE_GRAD_BATCHES \
        wandb.name="${STAGE1_EXP}"
fi

# Find Stage 1 checkpoint
# Hydra structure: outputs/YYYY-MM-DD/HH-MM-SS/outputs/experiment_name/
# For training (next stage): Use outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/last.ckpt
# For testing: Use outputs/YYYY-MM-DD/HH-MM-SS/outputs/experiment_name/lora_checkpoints/merged_final.ckpt
STAGE1_CKPT=""

# Method 1: Find experiment directory, then go to parent to find checkpoints
STAGE1_EXP_DIR=$(find outputs -type d -name "${STAGE1_EXP}" 2>/dev/null | sort -r | head -n1)
if [ -n "$STAGE1_EXP_DIR" ]; then
    # Go up two levels: outputs/experiment_name -> outputs/YYYY-MM-DD/HH-MM-SS/outputs -> outputs/YYYY-MM-DD/HH-MM-SS
    PARENT_DIR=$(dirname "$(dirname "$STAGE1_EXP_DIR")")
    if [ -d "${PARENT_DIR}/checkpoints" ]; then
        STAGE1_CKPT="${PARENT_DIR}/checkpoints/last.ckpt"
    fi
fi

# Method 2: Find by searching for last.ckpt in the same directory structure as experiment
if [ -z "$STAGE1_CKPT" ] || [ ! -f "$STAGE1_CKPT" ]; then
    # Find all experiment directories and check their parent directories
    for exp_dir in $(find outputs -type d -name "${STAGE1_EXP}" 2>/dev/null | sort -r); do
        parent_dir=$(dirname "$(dirname "$exp_dir")")
        candidate="${parent_dir}/checkpoints/last.ckpt"
        if [ -f "$candidate" ]; then
            STAGE1_CKPT="$candidate"
            break
        fi
    done
fi

# Method 3: Find most recent last.ckpt if still not found
if [ -z "$STAGE1_CKPT" ] || [ ! -f "$STAGE1_CKPT" ]; then
    STAGE1_CKPT=$(find outputs -name "last.ckpt" -type f 2>/dev/null | sort -r | head -n1)
    if [ -n "$STAGE1_CKPT" ]; then
        echo "⚠️  Warning: Using most recent checkpoint: $STAGE1_CKPT"
    fi
fi

# Convert to absolute path if relative
if [ -n "$STAGE1_CKPT" ] && [ ! "${STAGE1_CKPT:0:1}" = "/" ]; then
    STAGE1_CKPT="$(pwd)/${STAGE1_CKPT}"
fi

if [ -z "$STAGE1_CKPT" ] || [ ! -f "$STAGE1_CKPT" ]; then
    echo "❌ Error: Stage 1 checkpoint not found!"
    echo "   Searched for experiment: ${STAGE1_EXP}"
    if [ -n "$STAGE1_EXP_DIR" ]; then
        echo "   Found experiment directory: ${STAGE1_EXP_DIR}"
        PARENT_DIR=$(dirname "$(dirname "$STAGE1_EXP_DIR")")
        echo "   Expected checkpoint path: ${PARENT_DIR}/checkpoints/last.ckpt"
        echo "   Available files in ${PARENT_DIR}:"
        ls -la "${PARENT_DIR}" 2>/dev/null | head -10 || echo "   Directory not found"
    fi
    exit 1
fi
echo "✅ Stage 1 complete: $STAGE1_CKPT"

# ============================================================================
# Stage 2: Easy + Medium scenarios (with sampling weights)
# ============================================================================

echo ""
echo "Stage 2/3: Training on EASY + MEDIUM scenarios with weights [0.4, 0.4, 0.2] (rule-based score percentile)..."
STAGE2_EXP="${CURRICULUM_BASE_EXP}_stage2_mid"

# Stage 2: Combine stage1 (easy) + stage2 (medium) with weights [0.4, 0.4, 0.2]
# IMPORTANT: Using last.ckpt from Stage 1 (contains optimizer state for resume training)
# - last.ckpt: Training checkpoint with optimizer state, use for continuing training
# - merged_final.ckpt: Merged weights only, use for testing (NOT for training)
if [ "$LORA_ENABLED" = "true" ]; then
    python scripts/training/finetune_pluto.py \
        --config-name training/train_pluto_lora \
        experiment=$STAGE2_EXP \
        pretrained_ckpt=$STAGE1_CKPT \
        scenario_filter=$SCENARIO_FILTER_STAGE1 \
        +curriculum.splits=[$SCENARIO_FILTER_STAGE1,$SCENARIO_FILTER_STAGE2,$SCENARIO_FILTER_STAGE3] \
        +curriculum.sampling_weights=[0.01,0.98,0.01] \
        lora.enabled=true \
        lora.rank=$LORA_RANK \
        lora.alpha=$LORA_ALPHA \
        lora.dropout=$LORA_DROPOUT \
        lora.lora_lr=$LORA_LR \
        lora.policy_head_lr=$HEAD_LR \
        lora.ultra_minimal=$ULTRA_MINIMAL \
        +lora.is_curriculum_stage=true \
        lr=$LR \
        weight_decay=$WEIGHT_DECAY \
        epochs=$EPOCHS_STAGE2 \
        warmup_steps=$WARMUP_STEPS \
        data_loader.params.batch_size=$BATCH_SIZE \
        +trainer.params.accumulate_grad_batches=$ACCUMULATE_GRAD_BATCHES \
        wandb.name="${STAGE2_EXP}"
else
    python scripts/training/finetune_pluto.py \
        --config-name training/train_pluto_head_only \
        experiment=$STAGE2_EXP \
        pretrained_ckpt=$STAGE1_CKPT \
        scenario_filter=$SCENARIO_FILTER_STAGE2 \
        lora.enabled=false \
        lora.ultra_minimal=$ULTRA_MINIMAL \
        lora.policy_head_lr=$LR \
        lr=$LR \
        weight_decay=$WEIGHT_DECAY \
        epochs=$EPOCHS_STAGE2 \
        warmup_steps=$WARMUP_STEPS \
        data_loader.params.batch_size=$BATCH_SIZE \
        +trainer.params.accumulate_grad_batches=$ACCUMULATE_GRAD_BATCHES \
        wandb.name="${STAGE2_EXP}"
fi

# Find Stage 2 checkpoint
# Hydra structure: outputs/YYYY-MM-DD/HH-MM-SS/outputs/experiment_name/
# For training (next stage): Use outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/last.ckpt
# For testing: Use outputs/YYYY-MM-DD/HH-MM-SS/outputs/experiment_name/lora_checkpoints/merged_final.ckpt
STAGE2_CKPT=""

# Method 1: Find experiment directory, then go to parent to find checkpoints
STAGE2_EXP_DIR=$(find outputs -type d -name "${STAGE2_EXP}" 2>/dev/null | sort -r | head -n1)
if [ -n "$STAGE2_EXP_DIR" ]; then
    PARENT_DIR=$(dirname "$(dirname "$STAGE2_EXP_DIR")")
    if [ -d "${PARENT_DIR}/checkpoints" ]; then
        STAGE2_CKPT="${PARENT_DIR}/checkpoints/last.ckpt"
    fi
fi

# Method 2: Find by searching for last.ckpt in the same directory structure
if [ -z "$STAGE2_CKPT" ] || [ ! -f "$STAGE2_CKPT" ]; then
    for exp_dir in $(find outputs -type d -name "${STAGE2_EXP}" 2>/dev/null | sort -r); do
        parent_dir=$(dirname "$(dirname "$exp_dir")")
        candidate="${parent_dir}/checkpoints/last.ckpt"
        if [ -f "$candidate" ]; then
            STAGE2_CKPT="$candidate"
            break
        fi
    done
fi

# Method 3: Find most recent last.ckpt if still not found
if [ -z "$STAGE2_CKPT" ] || [ ! -f "$STAGE2_CKPT" ]; then
    STAGE2_CKPT=$(find outputs -name "last.ckpt" -type f 2>/dev/null | sort -r | head -n1)
    if [ -n "$STAGE2_CKPT" ]; then
        echo "⚠️  Warning: Using most recent checkpoint: $STAGE2_CKPT"
    fi
fi

# Convert to absolute path if relative
if [ -n "$STAGE2_CKPT" ] && [ ! "${STAGE2_CKPT:0:1}" = "/" ]; then
    STAGE2_CKPT="$(pwd)/${STAGE2_CKPT}"
fi

if [ -z "$STAGE2_CKPT" ] || [ ! -f "$STAGE2_CKPT" ]; then
    echo "❌ Error: Stage 2 checkpoint not found!"
    echo "   Searched for experiment: ${STAGE2_EXP}"
    if [ -n "$STAGE2_EXP_DIR" ]; then
        echo "   Found experiment directory: ${STAGE2_EXP_DIR}"
        PARENT_DIR=$(dirname "$(dirname "$STAGE2_EXP_DIR")")
        echo "   Expected checkpoint path: ${PARENT_DIR}/checkpoints/last.ckpt"
        echo "   Available files in ${PARENT_DIR}:"
        ls -la "${PARENT_DIR}" 2>/dev/null | head -10 || echo "   Directory not found"
    fi
    exit 1
fi
echo "✅ Stage 2 complete: $STAGE2_CKPT"

# ============================================================================
# Stage 3: Easy + Medium + Hard scenarios (with sampling weights)
# ============================================================================

echo ""
echo "Stage 3/3: Training on EASY + MEDIUM + HARD scenarios with weights [0.2, 0.3, 0.5] (rule-based score percentile)..."
STAGE3_EXP="${CURRICULUM_BASE_EXP}_stage3_high"

# Stage 3: Combine stage1 (easy) + stage2 (medium) + stage3 (hard) with weights [0.2, 0.3, 0.5]
# IMPORTANT: Using last.ckpt from Stage 2 (contains optimizer state for resume training)
# - last.ckpt: Training checkpoint with optimizer state, use for continuing training
# - merged_final.ckpt: Merged weights only, use for testing (NOT for training)
if [ "$LORA_ENABLED" = "true" ]; then
    python scripts/training/finetune_pluto.py \
        --config-name training/train_pluto_lora \
        experiment=$STAGE3_EXP \
        pretrained_ckpt=$STAGE2_CKPT \
        scenario_filter=$SCENARIO_FILTER_STAGE1 \
        +curriculum.splits=[$SCENARIO_FILTER_STAGE1,$SCENARIO_FILTER_STAGE2,$SCENARIO_FILTER_STAGE3] \
        +curriculum.sampling_weights=[0.01,0.01,0.98] \
        lora.enabled=true \
        lora.rank=$LORA_RANK \
        lora.alpha=$LORA_ALPHA \
        lora.dropout=$LORA_DROPOUT \
        lora.lora_lr=$LORA_LR \
        lora.policy_head_lr=$HEAD_LR \
        lora.ultra_minimal=$ULTRA_MINIMAL \
        +lora.is_curriculum_stage=true \
        lr=$LR \
        weight_decay=$WEIGHT_DECAY \
        epochs=$EPOCHS_STAGE3 \
        warmup_steps=$WARMUP_STEPS \
        data_loader.params.batch_size=$BATCH_SIZE \
        +trainer.params.accumulate_grad_batches=$ACCUMULATE_GRAD_BATCHES \
        wandb.name="${STAGE3_EXP}"
else
    python scripts/training/finetune_pluto.py \
        --config-name training/train_pluto_head_only \
        experiment=$STAGE3_EXP \
        pretrained_ckpt=$STAGE2_CKPT \
        scenario_filter=$SCENARIO_FILTER_STAGE3 \
        lora.enabled=false \
        lora.ultra_minimal=$ULTRA_MINIMAL \
        lora.policy_head_lr=$LR \
        lr=$LR \
        weight_decay=$WEIGHT_DECAY \
        epochs=$EPOCHS_STAGE3 \
        warmup_steps=$WARMUP_STEPS \
        data_loader.params.batch_size=$BATCH_SIZE \
        +trainer.params.accumulate_grad_batches=$ACCUMULATE_GRAD_BATCHES \
        wandb.name="${STAGE3_EXP}"
fi

# Find Stage 3 (final curriculum) checkpoint
# Hydra structure: outputs/YYYY-MM-DD/HH-MM-SS/outputs/experiment_name/
# For training (next stage): Use outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/last.ckpt
# For testing: Use outputs/YYYY-MM-DD/HH-MM-SS/outputs/experiment_name/lora_checkpoints/merged_final.ckpt
CURRICULUM_CKPT=""

# Method 1: Find experiment directory, then go to parent to find checkpoints
CURRICULUM_EXP_DIR=$(find outputs -type d -name "${STAGE3_EXP}" 2>/dev/null | sort -r | head -n1)
if [ -n "$CURRICULUM_EXP_DIR" ]; then
    PARENT_DIR=$(dirname "$(dirname "$CURRICULUM_EXP_DIR")")
    if [ -d "${PARENT_DIR}/checkpoints" ]; then
        CURRICULUM_CKPT="${PARENT_DIR}/checkpoints/last.ckpt"
    fi
fi

# Method 2: Find by searching for last.ckpt in the same directory structure
if [ -z "$CURRICULUM_CKPT" ] || [ ! -f "$CURRICULUM_CKPT" ]; then
    for exp_dir in $(find outputs -type d -name "${STAGE3_EXP}" 2>/dev/null | sort -r); do
        parent_dir=$(dirname "$(dirname "$exp_dir")")
        candidate="${parent_dir}/checkpoints/last.ckpt"
        if [ -f "$candidate" ]; then
            CURRICULUM_CKPT="$candidate"
            break
        fi
    done
fi

# Method 3: Find most recent last.ckpt if still not found
if [ -z "$CURRICULUM_CKPT" ] || [ ! -f "$CURRICULUM_CKPT" ]; then
    CURRICULUM_CKPT=$(find outputs -name "last.ckpt" -type f 2>/dev/null | sort -r | head -n1)
    if [ -n "$CURRICULUM_CKPT" ]; then
        echo "⚠️  Warning: Using most recent checkpoint: $CURRICULUM_CKPT"
    fi
fi

# Convert to absolute path if relative
if [ -n "$CURRICULUM_CKPT" ] && [ ! "${CURRICULUM_CKPT:0:1}" = "/" ]; then
    CURRICULUM_CKPT="$(pwd)/${CURRICULUM_CKPT}"
fi

if [ -z "$CURRICULUM_CKPT" ] || [ ! -f "$CURRICULUM_CKPT" ]; then
    echo "❌ Error: Curriculum checkpoint not found!"
    echo "   Searched for experiment: ${STAGE3_EXP}"
    if [ -n "$CURRICULUM_EXP_DIR" ]; then
        echo "   Found experiment directory: ${CURRICULUM_EXP_DIR}"
        PARENT_DIR=$(dirname "$(dirname "$CURRICULUM_EXP_DIR")")
        echo "   Expected checkpoint path: ${PARENT_DIR}/checkpoints/last.ckpt"
        echo "   Available files in ${PARENT_DIR}:"
        ls -la "${PARENT_DIR}" 2>/dev/null | head -10 || echo "   Directory not found"
    fi
    exit 1
fi
echo "✅ Curriculum checkpoint: $CURRICULUM_CKPT"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "=============================================="
echo "✅ ALL EXPERIMENTS COMPLETE! (Rule-based Score)"
echo "=============================================="
echo ""
echo "Results saved in outputs/"
echo ""
echo "Uniform fine-tuning:"
echo "  - Training: outputs/${UNIFORM_EXP}/"
echo "  - Checkpoint: ${UNIFORM_CKPT}"
if [ "$LORA_ENABLED" = "true" ]; then
    echo "  - LoRA checkpoint: ${UNIFORM_EXP_DIR}/lora_checkpoints/lora_final.pt"
    echo "  - Merged checkpoint: ${UNIFORM_EXP_DIR}/lora_checkpoints/merged_final.ckpt"
fi
echo ""
echo "Curriculum fine-tuning:"
echo "  - Stage 1 (easy): outputs/${CURRICULUM_BASE_EXP}_stage1_low/"
echo "  - Stage 2 (medium): outputs/${CURRICULUM_BASE_EXP}_stage2_mid/"
echo "  - Stage 3 (hard): outputs/${CURRICULUM_BASE_EXP}_stage3_high/"
echo "  - Final checkpoint: ${CURRICULUM_CKPT}"
if [ "$LORA_ENABLED" = "true" ]; then
    echo "  - LoRA checkpoint: ${CURRICULUM_EXP_DIR}/lora_checkpoints/lora_final.pt"
    echo "  - Merged checkpoint: ${CURRICULUM_EXP_DIR}/lora_checkpoints/merged_final.ckpt"
fi
echo ""
echo "Configuration used:"
echo "  LoRA enabled: $LORA_ENABLED"
if [ "$LORA_ENABLED" = "true" ]; then
    echo "  LoRA rank: $LORA_RANK, alpha: $LORA_ALPHA"
    echo "  LoRA LR: $LORA_LR, Head LR: $HEAD_LR"
fi
echo "  Epochs (uniform): $EPOCHS, Epochs per stage: $EPOCHS_STAGE1, $EPOCHS_STAGE2, $EPOCHS_STAGE3"
echo "  Weight decay: $WEIGHT_DECAY, Warmup steps: $WARMUP_STEPS"
echo "  Scenario filters: Rule-based score percentile splits"
echo ""
echo "Next: Analyze results and compare uniform vs curriculum training!"
echo "      Compare with LLM-based score results from run_lora_experiment.sh"
echo ""
