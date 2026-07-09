#!/bin/bash
# PLUTO LoRA fine-tuning experiment using frozen-PLUTO loss-ranked filters.
#
# The loss-ranked filters are generated from offline open-loop imitation losses:
#   llm-taxonomy/scripts/experiments/pluto/create_lossrank_filters.py

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
NUPLAN_DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-${WORKSPACE_ROOT}/nuplan-devkit}"
LOSSRANK_FILTER_DIR="${WORKSPACE_ROOT}/llm-taxonomy/artifacts/scenario_filters/pluto_lossrank"
PLUTO_FILTER_DIR="${REPO_ROOT}/config/scenario_filter"
cd "$REPO_ROOT"

# ============================================================================
# EASY PARAMETER TUNING - keep aligned with run_lora_experiment_llmbased.sh
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
SCENARIO_FILTER_STAGE1="lossrank_train_easy"
SCENARIO_FILTER_STAGE2="lossrank_train_medium"
SCENARIO_FILTER_STAGE3="lossrank_train_hard"
CURRICULUM_SPLITS="[$SCENARIO_FILTER_STAGE1,$SCENARIO_FILTER_STAGE2,$SCENARIO_FILTER_STAGE3]"
STAGE2_SAMPLING_WEIGHTS="[0.50,0.40,0.10]"
STAGE3_SAMPLING_WEIGHTS="[0.30,0.50,0.20]"
MAX_REPEAT_PER_SCENARIO="${MAX_REPEAT_PER_SCENARIO:-4}"
HARD_SUBTYPE_BALANCE="${HARD_SUBTYPE_BALANCE:-false}"

CURRICULUM_BASE_EXP="curriculum_lora_lossrank"

ensure_lossrank_filters() {
    local missing=0
    mkdir -p "$PLUTO_FILTER_DIR"

    for filter_name in "$SCENARIO_FILTER_UNIFORM" "$SCENARIO_FILTER_STAGE1" "$SCENARIO_FILTER_STAGE2" "$SCENARIO_FILTER_STAGE3"; do
        local target="${PLUTO_FILTER_DIR}/${filter_name}.yaml"
        local source="${LOSSRANK_FILTER_DIR}/${filter_name}.yaml"

        if [ ! -f "$target" ] && [ -f "$source" ]; then
            cp "$source" "$target"
            echo "Copied ${filter_name}.yaml into PLUTO config"
        fi

        if [ ! -f "$target" ]; then
            echo "Missing PLUTO scenario filter: $target"
            missing=1
        fi
    done

    if [ "$missing" -ne 0 ]; then
        echo ""
        echo "Generate loss-ranked filters first, for example:"
        echo "  cd ${WORKSPACE_ROOT}/pluto"
        echo "  python scripts/training/score_scenarios_by_loss.py \\"
        echo "    --config-name training/train_pluto_lora \\"
        echo "    scenario_filter=uniform_train_all \\"
        echo "    scenario_filter.remove_invalid_goals=false \\"
        echo "    +loss_scoring.output_path=${WORKSPACE_ROOT}/llm-taxonomy/artifacts/loss_scores/pluto_train_loss_scores.jsonl \\"
        echo "    +loss_scoring.rank_score=planning_loss \\"
        echo "    +loss_scoring.batch_size=$BATCH_SIZE"
        echo ""
        echo "  cd ${WORKSPACE_ROOT}/llm-taxonomy"
        echo "  python scripts/experiments/pluto/create_lossrank_filters.py \\"
        echo "    --loss-input artifacts/loss_scores/pluto_train_loss_scores.jsonl \\"
        echo "    --output-dir artifacts/scenario_filters/pluto_lossrank \\"
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

# Set up Python/runtime paths. Supports conda, .venv, or an already-active env.
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/env_bootstrap.sh"

ensure_lossrank_filters

echo "=============================================="
echo "PLUTO LoRA Fine-tuning Experiment (Loss-ranked)"
echo "=============================================="
echo "Curriculum filters: $SCENARIO_FILTER_STAGE1, $SCENARIO_FILTER_STAGE2, $SCENARIO_FILTER_STAGE3"
echo "Reference all-scenario filter: $SCENARIO_FILTER_UNIFORM"
echo "Experiment base: $CURRICULUM_BASE_EXP"
echo "Stage 1 distribution: uniform stabilization via $SCENARIO_FILTER_UNIFORM"
echo "Stage 2 sampling weights [easy,medium,hard]: $STAGE2_SAMPLING_WEIGHTS"
echo "Stage 3 sampling weights [easy,medium,hard]: $STAGE3_SAMPLING_WEIGHTS"
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
echo "EXPERIMENT: Curriculum fine-tuning (loss-ranked)"
echo "=============================================="

STAGE1_EXP="${CURRICULUM_BASE_EXP}_stage1_raw"
echo "Stage 1/3: uniform stabilization"
run_lora_train \
    "$STAGE1_EXP" \
    "pretrained_ckpt" \
    "$PRETRAINED_CKPT" \
    "$SCENARIO_FILTER_UNIFORM" \
    "$EPOCHS_STAGE1" \
    "$HEAD_LR1"
STAGE1_CKPT="$(find_latest_checkpoint "$STAGE1_EXP")"
echo "Stage 1 checkpoint: $STAGE1_CKPT"

STAGE2_EXP="${CURRICULUM_BASE_EXP}_stage2_mid"
echo "Stage 2/3: weighted loss-ranked curriculum"
run_lora_train \
    "$STAGE2_EXP" \
    "checkpoint" \
    "$STAGE1_CKPT" \
    "$SCENARIO_FILTER_STAGE1" \
    "$EPOCHS_STAGE2" \
    "$HEAD_LR2" \
    "+curriculum.splits=$CURRICULUM_SPLITS" \
    "+curriculum.sampling_weights=$STAGE2_SAMPLING_WEIGHTS" \
    "curriculum.score_method=loss" \
    "curriculum.bucket_split_rule=quantile_40_40_20" \
    "curriculum.max_repeat_per_scenario=$MAX_REPEAT_PER_SCENARIO" \
    "curriculum.hard_subtype_balance=$HARD_SUBTYPE_BALANCE" \
    "curriculum.sampling_log_path=artifacts/curriculum_sampling/${STAGE2_EXP}.json" \
    "curriculum.filter_file_path=${PLUTO_FILTER_DIR}/${SCENARIO_FILTER_STAGE1}.yaml"
STAGE2_CKPT="$(find_latest_checkpoint "$STAGE2_EXP")"
echo "Stage 2 checkpoint: $STAGE2_CKPT"

STAGE3_EXP="${CURRICULUM_BASE_EXP}_stage3_high"
echo "Stage 3/3: hard-weighted loss-ranked curriculum"
run_lora_train \
    "$STAGE3_EXP" \
    "checkpoint" \
    "$STAGE2_CKPT" \
    "$SCENARIO_FILTER_STAGE1" \
    "$EPOCHS_STAGE3" \
    "$HEAD_LR3" \
    "+curriculum.splits=$CURRICULUM_SPLITS" \
    "+curriculum.sampling_weights=$STAGE3_SAMPLING_WEIGHTS" \
    "curriculum.score_method=loss" \
    "curriculum.bucket_split_rule=quantile_40_40_20" \
    "curriculum.max_repeat_per_scenario=$MAX_REPEAT_PER_SCENARIO" \
    "curriculum.hard_subtype_balance=$HARD_SUBTYPE_BALANCE" \
    "curriculum.sampling_log_path=artifacts/curriculum_sampling/${STAGE3_EXP}.json" \
    "curriculum.filter_file_path=${PLUTO_FILTER_DIR}/${SCENARIO_FILTER_STAGE1}.yaml"
CURRICULUM_CKPT="$(find_latest_checkpoint "$STAGE3_EXP")"

echo ""
echo "=============================================="
echo "Done: Loss-ranked curriculum experiment complete"
echo "=============================================="
echo "Curriculum checkpoint: $CURRICULUM_CKPT"
echo "Use run_lora_experiment_uniform.sh for the uniform-principle curriculum baseline."
echo "Use run_lora_experiment_llmbased.sh for the LLM-guided curriculum experiment."
