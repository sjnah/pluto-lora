#!/bin/bash
################################################################################
# Quick Test: Configurable scenarios per group
# Validates that metrics are collected correctly
# Estimated time: 3-4 hours
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
NUPLAN_DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-${WORKSPACE_ROOT}/nuplan-devkit}"
INTERPLAN_ROOT="${WORKSPACE_ROOT}/interPlan"
cd "$REPO_ROOT"

# Configuration: Number of scenarios to evaluate per stage
# This ensures all enabled methods use the same scenarios
# Note: Sequential worker is used to minimize memory usage. 
# WARNING: If you get OOM (Out Of Memory) errors, enable batch processing (see BATCH_SIZE below).
# The simulation framework builds all simulation objects upfront, so large scenario counts can cause OOM.
SCENARIOS_PER_STAGE=${SCENARIOS_PER_STAGE:-2000} # 2000
FILTER_NAME=${FILTER_NAME:-val14_benchmark}
EXPERIMENT_SUFFIX=${EXPERIMENT_SUFFIX:-val14_benchmark}
TEST_LABEL=${TEST_LABEL:-Val14 Benchmark}
SCENARIO_BUILDER=${SCENARIO_BUILDER:-nuplan_v1_1_val}

# Model selection flags. Set any flag to false/0/no to skip that model.
RUN_ZERO_SHOT=${RUN_ZERO_SHOT:-false}
RUN_RULE_BASED=${RUN_RULE_BASED:-false}
RUN_LOSS_BASED=${RUN_LOSS_BASED:-false}
RUN_UNIFORM=${RUN_UNIFORM:-false}
RUN_RANDOM_BUCKET=${RUN_RANDOM_BUCKET:-false}
RUN_LLM_CURRICULUM=${RUN_LLM_CURRICULUM:-false}
RUN_MPOC=${RUN_MPOC:-false}

LLM_CURRICULUM_VERSION=${LLM_CURRICULUM_VERSION:-v2}
LLM_CURRICULUM_SLUG=${LLM_CURRICULUM_SLUG:-curriculum_llm_guided_${LLM_CURRICULUM_VERSION}}
LLM_CURRICULUM_EXP=${LLM_CURRICULUM_EXP:-curriculum_lora_llm_guided_${LLM_CURRICULUM_VERSION}_stage3_high}

# Batch size for processing scenarios (to avoid OOM)
# If set to a positive number, scenarios will be automatically split into batches and processed sequentially.
# Example: BATCH_SIZE=200 will process scenarios in batches of 200.
# Set to empty or 0 to disable batching (not recommended for large scenario counts).
# Recommended: 150-300 depending on available memory
BATCH_SIZE=${BATCH_SIZE:-100}

# Helper function to run simulation (with automatic batching if enabled)
run_simulation() {
    local filter=$1
    local ckpt=$2
    local experiment=$3
    local scenario_builder=${4:-""}
    
    if [ -n "$BATCH_SIZE" ] && [ "$BATCH_SIZE" -gt 0 ]; then
        local builder_arg=""
        [ -n "$scenario_builder" ] && builder_arg="--scenario-builder $scenario_builder"
        
        python ${REPO_ROOT}/scripts/evaluation/run_simulation_batched.py \
            --filter "$filter" \
            --ckpt "$ckpt" \
            --experiment "$experiment" \
            --batch-size $BATCH_SIZE \
            --limit $SCENARIOS_PER_STAGE \
            $builder_arg
    else
        local builder_arg=""
        [ -n "$scenario_builder" ] && builder_arg="scenario_builder=$scenario_builder"
        
        python -X faulthandler ${REPO_ROOT}/run_simulation.py \
            +simulation=closed_loop_nonreactive_agents \
            observation=box_observation \
            ego_controller=two_stage_controller \
            planner=pluto_planner \
            +planner.pluto_planner.planner_ckpt="$ckpt" \
            scenario_filter="$filter" \
            scenario_filter.limit_total_scenarios=$SCENARIOS_PER_STAGE \
            $builder_arg \
            experiment="$experiment" \
            worker=sequential
    fi
}

is_enabled() {
    case "$1" in
        true|TRUE|True|1|yes|YES|Yes|on|ON|On) return 0 ;;
        *) return 1 ;;
    esac
}

count_enabled_models() {
    local count=0
    is_enabled "$RUN_ZERO_SHOT" && count=$((count + 1))
    is_enabled "$RUN_RULE_BASED" && count=$((count + 1))
    is_enabled "$RUN_LOSS_BASED" && count=$((count + 1))
    is_enabled "$RUN_UNIFORM" && count=$((count + 1))
    is_enabled "$RUN_RANDOM_BUCKET" && count=$((count + 1))
    is_enabled "$RUN_LLM_CURRICULUM" && count=$((count + 1))
    is_enabled "$RUN_MPOC" && count=$((count + 1))
    echo "$count"
}

find_lora_checkpoint() {
    local result_var=$1
    local label=$2
    local experiment_name=$3

    local exp_dir
    exp_dir=$(find outputs -type d -name "$experiment_name" 2>/dev/null | sort -r | head -n1)

    if [ -z "$exp_dir" ]; then
        echo "Error: ${label} LoRA training output directory not found!"
        echo ""
        echo "   Searched for: ${experiment_name}"
        echo ""
        echo "   Available related experiments:"
        find outputs -type d -name "*lora*" 2>/dev/null | head -10
        exit 1
    fi

    local ckpt="${exp_dir}/lora_checkpoints/merged_final.ckpt"
    if [ ! -f "$ckpt" ]; then
        ckpt="${exp_dir}/checkpoints/last.ckpt"
    fi

    if [ ! "${ckpt:0:1}" = "/" ]; then
        ckpt="$(pwd)/${ckpt}"
    fi

    if [ ! -f "$ckpt" ]; then
        echo "Error: ${label} checkpoint not found!"
        echo "   Tried: ${exp_dir}/lora_checkpoints/merged_final.ckpt"
        echo "   Tried: ${exp_dir}/checkpoints/last.ckpt"
        echo "   Available files:"
        ls -la "${exp_dir}/lora_checkpoints/" 2>/dev/null || echo "   lora_checkpoints directory not found"
        ls -la "${exp_dir}/checkpoints/" 2>/dev/null || echo "   checkpoints directory not found"
        exit 1
    fi

    printf -v "$result_var" '%s' "$ckpt"
}

run_model_simulation() {
    local label=$1
    local slug=$2
    local ckpt=$3
    local filter=$4
    local experiment_suffix=$5
    local scenario_builder=${6:-""}
    local run_mode=${7:-batched}
    local experiment="quick_test_${slug}_${experiment_suffix}"

    echo ""
    echo "Running ${label} on ${filter}..."

    if [ "$run_mode" = "direct_thread_pool" ]; then
        python ${REPO_ROOT}/run_simulation.py \
            +simulation=closed_loop_nonreactive_agents \
            observation=box_observation \
            ego_controller=two_stage_controller \
            planner=pluto_planner \
            +planner.pluto_planner.planner_ckpt="$ckpt" \
            scenario_filter="$filter" \
            scenario_filter.limit_total_scenarios=$SCENARIOS_PER_STAGE \
            experiment="$experiment" \
            worker=single_machine_thread_pool \
            worker.max_workers=1
    else
        run_simulation "$filter" "$ckpt" "$experiment" "$scenario_builder"
    fi

    local metrics_dir="${NUPLAN_EXP_ROOT}/exp/${experiment}/metrics"
    local record_file="${SCENARIO_RECORDS_DIR}/${experiment}.json"
    python ${REPO_ROOT}/scripts/evaluation/save_scenario_tokens.py "$metrics_dir" "$record_file" || echo "Could not save scenario tokens"

    echo "${label} ${filter} done!"
}

run_enabled_models() {
    local filter=$1
    local experiment_suffix=$2
    local scenario_builder=${3:-""}
    local run_mode=${4:-batched}

    is_enabled "$RUN_ZERO_SHOT" && run_model_simulation "Zero-shot" "zeroshot" "$ZERO_SHOT_CKPT" "$filter" "$experiment_suffix" "$scenario_builder" "$run_mode"
    is_enabled "$RUN_RULE_BASED" && run_model_simulation "Rule-based" "rulebased" "$RULE_BASED_CKPT" "$filter" "$experiment_suffix" "$scenario_builder" "$run_mode"
    is_enabled "$RUN_LOSS_BASED" && run_model_simulation "Loss-based" "lossbased" "$LOSS_BASED_CKPT" "$filter" "$experiment_suffix" "$scenario_builder" "$run_mode"
    is_enabled "$RUN_UNIFORM" && run_model_simulation "Uniform curriculum" "curriculum_uniform" "$UNIFORM_CKPT" "$filter" "$experiment_suffix" "$scenario_builder" "$run_mode"
    is_enabled "$RUN_RANDOM_BUCKET" && run_model_simulation "RandomBucket-FT" "curriculum_randombucket" "$RANDOM_BUCKET_CKPT" "$filter" "$experiment_suffix" "$scenario_builder" "$run_mode"
    is_enabled "$RUN_LLM_CURRICULUM" && run_model_simulation "LLM-guided curriculum (${LLM_CURRICULUM_VERSION})" "$LLM_CURRICULUM_SLUG" "$CURRICULUM_CKPT" "$filter" "$experiment_suffix" "$scenario_builder" "$run_mode"
    is_enabled "$RUN_MPOC" && run_model_simulation "MPOC curriculum" "curriculum_mpoc" "$MPOC_CKPT" "$filter" "$experiment_suffix" "$scenario_builder" "$run_mode"
}

# Set up Python/runtime paths. Supports conda, .venv, or an already-active env.
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/env_bootstrap.sh"

# Directory to save scenario token records
SCENARIO_RECORDS_DIR="${REPO_ROOT}/artifacts/records/scenario_records"
mkdir -p "$SCENARIO_RECORDS_DIR"

ENABLED_MODEL_COUNT=$(count_enabled_models)
if [ "$ENABLED_MODEL_COUNT" -eq 0 ]; then
    echo "Error: All model flags are disabled. Enable at least one model."
    exit 1
fi

SCENARIO_GROUP_COUNT=4
TOTAL_SCENARIOS=$((SCENARIOS_PER_STAGE * SCENARIO_GROUP_COUNT * ENABLED_MODEL_COUNT))

echo "=============================================="
echo "Quick Test: ${TOTAL_SCENARIOS} scenario executions (${SCENARIOS_PER_STAGE} per group, ${ENABLED_MODEL_COUNT} enabled models)"
echo "Validating metric collection"
echo "=============================================="
echo ""

# Find checkpoints
if is_enabled "$RUN_ZERO_SHOT"; then
    ZERO_SHOT_CKPT="$(pwd)/checkpoints/pluto_1M_aux_cil.ckpt"
    if [ ! -f "$ZERO_SHOT_CKPT" ]; then
        echo "Error: Zero-shot checkpoint not found: $ZERO_SHOT_CKPT"
        exit 1
    fi
fi

is_enabled "$RUN_RULE_BASED" && find_lora_checkpoint RULE_BASED_CKPT "Rule-based" "curriculum_lora_rulebased_stage3_high"
is_enabled "$RUN_LOSS_BASED" && find_lora_checkpoint LOSS_BASED_CKPT "Loss-based" "curriculum_lora_lossrank_stage3_high"
is_enabled "$RUN_UNIFORM" && find_lora_checkpoint UNIFORM_CKPT "Uniform curriculum" "curriculum_lora_uniform"
is_enabled "$RUN_RANDOM_BUCKET" && find_lora_checkpoint RANDOM_BUCKET_CKPT "RandomBucket-FT" "curriculum_lora_randombucket_stage3_high"
is_enabled "$RUN_LLM_CURRICULUM" && find_lora_checkpoint CURRICULUM_CKPT "LLM-guided curriculum" "$LLM_CURRICULUM_EXP"
is_enabled "$RUN_MPOC" && find_lora_checkpoint MPOC_CKPT "MPOC curriculum" "curriculum_lora_mpoc_stage3_high"

echo "📍 Using checkpoints:"
is_enabled "$RUN_ZERO_SHOT" && echo "  Zero-shot:       $ZERO_SHOT_CKPT (PLUTO, no fine-tuning)"
is_enabled "$RUN_RULE_BASED" && echo "  Rule-based:      $RULE_BASED_CKPT (PLUTO + rule-based curriculum LoRA)"
is_enabled "$RUN_LOSS_BASED" && echo "  Loss-based:      $LOSS_BASED_CKPT (PLUTO + loss-ranked curriculum LoRA)"
is_enabled "$RUN_UNIFORM" && echo "  Uniform:         $UNIFORM_CKPT (PLUTO + uniform-principle curriculum LoRA)"
is_enabled "$RUN_RANDOM_BUCKET" && echo "  RandomBucket-FT: $RANDOM_BUCKET_CKPT (PLUTO + random-bucket curriculum LoRA)"
is_enabled "$RUN_LLM_CURRICULUM" && echo "  LLM-guided:      $CURRICULUM_CKPT (PLUTO + ${LLM_CURRICULUM_VERSION} curriculum LoRA, slug=${LLM_CURRICULUM_SLUG})"
is_enabled "$RUN_MPOC" && echo "  MPOC curriculum: $MPOC_CKPT (PLUTO + MPOC curriculum LoRA)"
echo ""

START_TIME=$(date +%s)

################################################################################
# Test 1: Val14 benchmark
################################################################################
echo ""
echo "=============================================="
echo "Testing ${TEST_LABEL} - ${SCENARIOS_PER_STAGE} scenarios"
echo "=============================================="
run_enabled_models "$FILTER_NAME" "$EXPERIMENT_SUFFIX" "$SCENARIO_BUILDER"

################################################################################
# Test 2: Easy scenarios
################################################################################
echo ""
echo "=============================================="
echo "Testing Easy (llm_guided_val) - ${SCENARIOS_PER_STAGE} scenarios"
echo "=============================================="
#run_enabled_models llm_guided_val_easy llm_guided_val_easy "" direct_thread_pool

################################################################################
# Test 3: Medium scenarios
################################################################################
echo ""
echo "=============================================="
echo "Testing Medium (llm_guided_val) - ${SCENARIOS_PER_STAGE} scenarios"
echo "=============================================="
#run_enabled_models llm_guided_val_medium llm_guided_val_medium

################################################################################
# Test 4: Hard scenarios
################################################################################
echo ""
echo "=============================================="
echo "Testing Hard (llm_guided_val) - ${SCENARIOS_PER_STAGE} scenarios"
echo "=============================================="
#run_enabled_models llm_guided_val_hard llm_guided_val_hard

################################################################################
# Summary and Analysis
################################################################################

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=============================================="
echo "✅ Quick test complete!"
echo "=============================================="
echo "Time taken: ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results are in: ${NUPLAN_EXP_ROOT}/exp/quick_test_*"
echo ""
echo "Analyzing results..."
python ${REPO_ROOT}/scripts/analysis/analyze_quick_test.py

echo ""
echo "=============================================="
echo "Next steps:"
echo "  1. Check if NR-CLS metrics are present"
echo "  2. If OK, use bash analyze.sh for later analysis reruns"
echo "  3. If not OK, debug metric configuration"
echo "=============================================="
