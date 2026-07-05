#!/bin/bash
################################################################################
# Quick Test: Test14-Hard Filter
# Uses test14-hard.yaml scenario filter with nuplan-v1.1_test dataset
# Validates that metrics are collected correctly
#
# Simulation Type:
#   - Default: reactive (closed_loop_reactive_agents) - More realistic, other vehicles react to ego
#   - Alternative: nonreactive (closed_loop_nonreactive_agents) - Other vehicles follow fixed trajectories
#
# Usage:
#   ./quick_test_test14.sh                    # Uses reactive (default)
#   SIMULATION_TYPE=nonreactive ./quick_test_test14.sh  # Use nonreactive
#   SIMULATION_TYPE=reactive ./quick_test_test14.sh     # Explicitly use reactive
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
NUPLAN_DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-${WORKSPACE_ROOT}/nuplan-devkit}"
INTERPLAN_ROOT="${WORKSPACE_ROOT}/interPlan"
cd "$REPO_ROOT"

FILTER_NAME=${FILTER_NAME:-test14-hard}
EXPERIMENT_SUFFIX=${EXPERIMENT_SUFFIX:-test14_hard}
TEST_LABEL=${TEST_LABEL:-Test14-Hard}
SCENARIO_BUILDER=${SCENARIO_BUILDER:-nuplan_v1_1_test}
COLLECT_TEST=${COLLECT_TEST:-test14-hard}

# Configuration: Number of scenarios to evaluate per stage
# This ensures all enabled methods use the same scenarios
# Note: Sequential worker is used to minimize memory usage. 
# WARNING: If you get OOM (Out Of Memory) errors, enable batch processing (see BATCH_SIZE below).
# The simulation framework builds all simulation objects upfront, so large scenario counts can cause OOM.
# test14-hard.yaml contains 286 scenarios, so default to using all of them
SCENARIOS_PER_STAGE=${SCENARIOS_PER_STAGE:-286} # 286

# Model selection flags. Set any flag to false/0/no to skip that model.
RUN_ZERO_SHOT=${RUN_ZERO_SHOT:-false}
RUN_RULE_BASED=${RUN_RULE_BASED:-false}
RUN_LOSS_BASED=${RUN_LOSS_BASED:-false}
RUN_UNIFORM=${RUN_UNIFORM:-false}
RUN_RANDOM_BUCKET=${RUN_RANDOM_BUCKET:-false}
RUN_LLM_CURRICULUM=${RUN_LLM_CURRICULUM:-false}
RUN_MPOC=${RUN_MPOC:-false}

# Batch size for processing scenarios (to avoid OOM)
# If set to a positive number, scenarios will be automatically split into batches and processed sequentially.
# Example: BATCH_SIZE=200 will process scenarios in batches of 200.
# Set to empty or 0 to disable batching (not recommended for large scenario counts).
# Recommended: 150-300 depending on available memory
BATCH_SIZE=${BATCH_SIZE:-50}

# Simulation type: reactive or nonreactive
# For test datasets, reactive agents provide more realistic evaluation as other vehicles react to ego actions
# Set to "reactive" for closed_loop_reactive_agents or "nonreactive" for closed_loop_nonreactive_agents
SIMULATION_TYPE=${SIMULATION_TYPE:-reactive}

# Console progress controls. The direct reactive path needs verbose=true
# explicitly; otherwise nuPlan's sequential worker progress bar is disabled.
SIMULATION_VERBOSE=${SIMULATION_VERBOSE:-true}
ENABLE_PROGRESS_BAR=${ENABLE_PROGRESS_BAR:-true}

# Helper function to run simulation (with automatic batching if enabled)
run_simulation() {
    local filter=$1
    local ckpt=$2
    local experiment=$3
    local scenario_builder=${4:-""}
    
    # Determine simulation type
    if [ "$SIMULATION_TYPE" = "reactive" ]; then
        local simulation_config="closed_loop_reactive_agents"
        local observation_config="idm_agents_observation"
    else
        local simulation_config="closed_loop_nonreactive_agents"
        local observation_config="box_observation"
    fi
    
    if [ -n "$BATCH_SIZE" ] && [ "$BATCH_SIZE" -gt 0 ] && [ "$SIMULATION_TYPE" != "reactive" ]; then
        # Batching is only supported for nonreactive agents
        # run_simulation_batched.py hardcodes nonreactive agents
        local builder_arg=""
        [ -n "$scenario_builder" ] && builder_arg="--scenario-builder $scenario_builder"
        
        python ${SCRIPT_DIR}/run_simulation_batched.py \
            --filter "$filter" \
            --ckpt "$ckpt" \
            --experiment "$experiment" \
            --batch-size $BATCH_SIZE \
            --limit $SCENARIOS_PER_STAGE \
            --simulation-verbose "$SIMULATION_VERBOSE" \
            $builder_arg
    else
        # Run directly (either batching disabled, reactive agents, or batching not supported)
        if [ -n "$BATCH_SIZE" ] && [ "$BATCH_SIZE" -gt 0 ] && [ "$SIMULATION_TYPE" = "reactive" ]; then
            echo "⚠️  Warning: Batching is not yet supported for reactive agents."
            echo "   Running without batching..."
        fi
        
        local builder_arg=""
        [ -n "$scenario_builder" ] && builder_arg="scenario_builder=$scenario_builder"
        
        python -X faulthandler ${REPO_ROOT}/run_simulation.py \
            +simulation=$simulation_config \
            observation=$observation_config \
            ego_controller=two_stage_controller \
            planner=pluto_planner \
            +planner.pluto_planner.planner_ckpt="$ckpt" \
            scenario_filter="$filter" \
            +scenario_filter.limit_total_scenarios=$SCENARIOS_PER_STAGE \
            verbose="$SIMULATION_VERBOSE" \
            enable_simulation_progress_bar="$ENABLE_PROGRESS_BAR" \
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
    local experiment="quick_test_${slug}_${experiment_suffix}"

    echo ""
    echo "Running ${label} on ${filter}..."
    run_simulation "$filter" "$ckpt" "$experiment" "$scenario_builder"

    local metrics_dir="${NUPLAN_EXP_ROOT}/exp/${experiment}/metrics"
    local record_file="${SCENARIO_RECORDS_DIR}/${experiment}.json"
    python ${REPO_ROOT}/scripts/evaluation/save_scenario_tokens.py "$metrics_dir" "$record_file" || echo "Could not save scenario tokens"

    echo "${label} ${filter} done!"
}

run_enabled_models() {
    local filter=$1
    local experiment_suffix=$2
    local scenario_builder=${3:-""}

    is_enabled "$RUN_ZERO_SHOT" && run_model_simulation "Zero-shot" "zeroshot" "$ZERO_SHOT_CKPT" "$filter" "$experiment_suffix" "$scenario_builder"
    is_enabled "$RUN_RULE_BASED" && run_model_simulation "Rule-based" "rulebased" "$RULE_BASED_CKPT" "$filter" "$experiment_suffix" "$scenario_builder"
    is_enabled "$RUN_LOSS_BASED" && run_model_simulation "Loss-based" "lossbased" "$LOSS_BASED_CKPT" "$filter" "$experiment_suffix" "$scenario_builder"
    is_enabled "$RUN_UNIFORM" && run_model_simulation "Uniform curriculum" "curriculum_uniform" "$UNIFORM_CKPT" "$filter" "$experiment_suffix" "$scenario_builder"
    is_enabled "$RUN_RANDOM_BUCKET" && run_model_simulation "RandomBucket-FT" "curriculum_randombucket" "$RANDOM_BUCKET_CKPT" "$filter" "$experiment_suffix" "$scenario_builder"
    is_enabled "$RUN_LLM_CURRICULUM" && run_model_simulation "LLM-based curriculum" "curriculum_llmbased" "$CURRICULUM_CKPT" "$filter" "$experiment_suffix" "$scenario_builder"
    is_enabled "$RUN_MPOC" && run_model_simulation "MPOC curriculum" "curriculum_mpoc" "$MPOC_CKPT" "$filter" "$experiment_suffix" "$scenario_builder"
}

# Set up Python/runtime paths. Supports conda, .venv, or an already-active env.
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/env_bootstrap.sh"

# Directory to save scenario token records
SCENARIO_RECORDS_DIR="${REPO_ROOT}/artifacts/records/scenario_records"
mkdir -p "$SCENARIO_RECORDS_DIR"

if [ ! -f "${REPO_ROOT}/config/scenario_filter/${FILTER_NAME}.yaml" ]; then
    echo "Error: scenario filter not found: ${REPO_ROOT}/config/scenario_filter/${FILTER_NAME}.yaml"
    exit 1
fi

if [ "$SCENARIOS_PER_STAGE" = "auto" ]; then
    SCENARIOS_PER_STAGE=$(python - "$REPO_ROOT/config/scenario_filter/${FILTER_NAME}.yaml" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
in_tokens = False
count = 0
for line in path.read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    if stripped == "scenario_tokens:":
        in_tokens = True
        continue
    if in_tokens and stripped and not stripped.startswith("-"):
        break
    if in_tokens and stripped.startswith("-"):
        count += 1
print(count)
PY
)
fi

ENABLED_MODEL_COUNT=$(count_enabled_models)
if [ "$ENABLED_MODEL_COUNT" -eq 0 ]; then
    echo "Error: All model flags are disabled. Enable at least one model."
    exit 1
fi

TOTAL_SCENARIOS=$((SCENARIOS_PER_STAGE * ENABLED_MODEL_COUNT))

echo "=============================================="
echo "Quick Test (${TEST_LABEL}): ${TOTAL_SCENARIOS} scenario executions (${SCENARIOS_PER_STAGE} per method)"
echo "Using ${FILTER_NAME}.yaml filter with ${SCENARIO_BUILDER} dataset"
echo "Simulation type: ${SIMULATION_TYPE} (other vehicles ${SIMULATION_TYPE} to ego)"
echo "Simulation verbose: ${SIMULATION_VERBOSE}"
echo "Progress bar: ${ENABLE_PROGRESS_BAR}"
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
is_enabled "$RUN_LLM_CURRICULUM" && find_lora_checkpoint CURRICULUM_CKPT "LLM-based curriculum" "curriculum_lora_llmbased_stage3_high"
is_enabled "$RUN_MPOC" && find_lora_checkpoint MPOC_CKPT "MPOC curriculum" "curriculum_lora_mpoc_stage3_high"

echo "📍 Using checkpoints:"
is_enabled "$RUN_ZERO_SHOT" && echo "  Zero-shot:       $ZERO_SHOT_CKPT (PLUTO, no fine-tuning)"
is_enabled "$RUN_RULE_BASED" && echo "  Rule-based:      $RULE_BASED_CKPT (PLUTO + rule-based curriculum LoRA)"
is_enabled "$RUN_LOSS_BASED" && echo "  Loss-based:      $LOSS_BASED_CKPT (PLUTO + loss-ranked curriculum LoRA)"
is_enabled "$RUN_UNIFORM" && echo "  Uniform:         $UNIFORM_CKPT (PLUTO + uniform-principle curriculum LoRA)"
is_enabled "$RUN_RANDOM_BUCKET" && echo "  RandomBucket-FT: $RANDOM_BUCKET_CKPT (PLUTO + random-bucket curriculum LoRA)"
is_enabled "$RUN_LLM_CURRICULUM" && echo "  LLM curriculum:  $CURRICULUM_CKPT (PLUTO + LLM-based curriculum LoRA)"
is_enabled "$RUN_MPOC" && echo "  MPOC curriculum: $MPOC_CKPT (PLUTO + MPOC curriculum LoRA)"
echo ""
echo "📍 Using scenario filter: ${FILTER_NAME}"
echo "📍 Using scenario builder: ${SCENARIO_BUILDER}"
echo ""

START_TIME=$(date +%s)

################################################################################
# Test: Test14-Hard scenarios
################################################################################
echo ""
echo "=============================================="
echo "Testing ${TEST_LABEL} - ${SCENARIOS_PER_STAGE} scenarios"
echo "=============================================="
run_enabled_models "$FILTER_NAME" "$EXPERIMENT_SUFFIX" "$SCENARIO_BUILDER"

################################################################################
# Summary and Analysis
################################################################################

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=============================================="
echo "✅ Quick test (${FILTER_NAME}) complete!"
echo "=============================================="
echo "Time taken: ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results are in: ${NUPLAN_EXP_ROOT}/exp/quick_test_*_${EXPERIMENT_SUFFIX}"
echo ""
echo "Collecting result summary..."
python ${REPO_ROOT}/scripts/evaluation/collect_quick_test_results.py \
    --tests "$COLLECT_TEST" \
    --methods zeroshot,lossbased,curriculum_randombucket,curriculum_llmbased,curriculum_mpoc \
    --detail || echo "Could not collect ${FILTER_NAME} summary"

echo ""
echo "=============================================="
echo "Next steps:"
echo "  1. Check if NR-CLS metrics are present"
echo "  2. Compare results with other test datasets"
echo "  3. Verify all scenarios were processed correctly"
echo "=============================================="
