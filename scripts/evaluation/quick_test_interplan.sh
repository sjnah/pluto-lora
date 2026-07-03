#!/bin/bash
################################################################################
# Quick Test: InterPlan Benchmark
# Uses interPlan's official scenario filters (interplan10 or benchmark_scenarios)
# Validates that metrics are collected correctly
#
# This script runs interPlan benchmark with official filters
# for enabled methods (zero-shot, rule-based, loss-based, uniform, random-bucket, LLM curriculum)
#
# Usage:
#   ./quick_test_interplan.sh [interplan10|benchmark_scenarios]
#   Default: interplan10 (80 scenarios, 10 per type)
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
NUPLAN_DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-${WORKSPACE_ROOT}/nuplan-devkit}"
INTERPLAN_ROOT="${WORKSPACE_ROOT}/interPlan"
cd "$REPO_ROOT"

# Configuration: InterPlan scenario filter
# - interplan10: Official benchmark with 80 scenarios (10 per type)
# - benchmark_scenarios: All 335 scenarios
INTERPLAN_FILTER=${1:-interplan10}

# Model selection flags. Set any flag to false/0/no to skip that model.
RUN_ZERO_SHOT=${RUN_ZERO_SHOT:-true}
RUN_RULE_BASED=${RUN_RULE_BASED:-true}
RUN_LOSS_BASED=${RUN_LOSS_BASED:-true}
RUN_UNIFORM=${RUN_UNIFORM:-true}
RUN_RANDOM_BUCKET=${RUN_RANDOM_BUCKET:-true}
RUN_LLM_CURRICULUM=${RUN_LLM_CURRICULUM:-true}

if [ "$INTERPLAN_FILTER" != "interplan10" ] && [ "$INTERPLAN_FILTER" != "benchmark_scenarios" ]; then
    echo "❌ Error: Invalid filter. Use 'interplan10' or 'benchmark_scenarios'"
    exit 1
fi

# InterPlan paths
INTERPLAN_SCRIPT="${INTERPLAN_ROOT}/interplan/planning/script/run_simulation.py"

if [ ! -f "$INTERPLAN_SCRIPT" ]; then
    echo "❌ Error: InterPlan script not found: $INTERPLAN_SCRIPT"
    exit 1
fi

# Helper function to run interPlan simulation
# Follows interPlan's official example (sim_pdm_closed.sh)
run_interplan_simulation() {
    local filter=$1
    local ckpt=$2
    local experiment=$3
    
    echo "   Running interPlan simulation with checkpoint: $ckpt"
    
    # Use interPlan's official format (following sim_pdm_closed.sh example)
    # Add pluto config directory to searchpath so pluto_planner can be found
    # Override scenario_builder data_root to point to correct location
    # interPlan's default interplan.yaml points to trainval which may not exist
    # We need to find where the actual data is located
    python "$INTERPLAN_SCRIPT" \
        +simulation=default_interplan_benchmark \
        scenario_filter=$filter \
        scenario_builder.data_root='${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1_test/data/cache/test' \
        scenario_builder.sensor_root='${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1_test/sensor_blobs' \
        planner=pluto_planner \
        +planner.pluto_planner.planner_ckpt="$ckpt" \
        experiment_name="$experiment" \
        hydra.searchpath="[\
pkg://interplan.planning.script.config.common,\
pkg://interplan.planning.script.config.simulation,\
pkg://interplan.planning.script.experiments,\
pkg://tuplan_garage.planning.script.config.common,\
pkg://tuplan_garage.planning.script.config.simulation,\
pkg://nuplan.planning.script.config.common,\
pkg://nuplan.planning.script.config.simulation,\
pkg://nuplan.planning.script.experiments,\
${REPO_ROOT}/config\
]"
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
        exit 1
    fi

    printf -v "$result_var" '%s' "$ckpt"
}

run_model_interplan_simulation() {
    local label=$1
    local slug=$2
    local ckpt=$3
    local experiment_suffix=$4
    local experiment="quick_test_interplan_${slug}_${experiment_suffix}"

    echo ""
    echo "Running ${label} on interPlan (${INTERPLAN_FILTER})..."
    run_interplan_simulation "$INTERPLAN_FILTER" "$ckpt" "$experiment"

    local metrics_dir="${NUPLAN_EXP_ROOT}/exp/${experiment}"
    local record_file="${SCENARIO_RECORDS_DIR}/${experiment}.json"
    if [ -f "${REPO_ROOT}/scripts/evaluation/save_scenario_tokens.py" ]; then
        python ${REPO_ROOT}/scripts/evaluation/save_scenario_tokens.py "$metrics_dir" "$record_file" || echo "Could not save scenario tokens"
    fi

    echo "${label} interPlan (${INTERPLAN_FILTER}) done!"
}

run_enabled_interplan_models() {
    local experiment_suffix=$1

    is_enabled "$RUN_ZERO_SHOT" && run_model_interplan_simulation "Zero-shot" "zeroshot" "$ZERO_SHOT_CKPT" "$experiment_suffix"
    is_enabled "$RUN_RULE_BASED" && run_model_interplan_simulation "Rule-based" "rulebased" "$RULE_BASED_CKPT" "$experiment_suffix"
    is_enabled "$RUN_LOSS_BASED" && run_model_interplan_simulation "Loss-based" "lossbased" "$LOSS_BASED_CKPT" "$experiment_suffix"
    is_enabled "$RUN_UNIFORM" && run_model_interplan_simulation "Uniform curriculum" "curriculum_uniform" "$UNIFORM_CKPT" "$experiment_suffix"
    is_enabled "$RUN_RANDOM_BUCKET" && run_model_interplan_simulation "RandomBucket-FT" "curriculum_randombucket" "$RANDOM_BUCKET_CKPT" "$experiment_suffix"
    is_enabled "$RUN_LLM_CURRICULUM" && run_model_interplan_simulation "LLM-based curriculum" "curriculum_llmbased" "$CURRICULUM_CKPT" "$experiment_suffix"
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

if [ "$INTERPLAN_FILTER" = "interplan10" ]; then
    TOTAL_SCENARIOS=$((80 * ENABLED_MODEL_COUNT))
else
    TOTAL_SCENARIOS=$((335 * ENABLED_MODEL_COUNT))
fi

echo "=============================================="
echo "Quick Test (InterPlan Benchmark)"
echo "Using interPlan's official filter: $INTERPLAN_FILTER"
if [ "$INTERPLAN_FILTER" = "interplan10" ]; then
    echo "  - 80 scenarios (10 per type)"
    echo "  - Official benchmark used in paper"
elif [ "$INTERPLAN_FILTER" = "benchmark_scenarios" ]; then
    echo "  - 335 scenarios (all available)"
fi
echo "Enabled models: ${ENABLED_MODEL_COUNT}"
echo "Scenario executions: ${TOTAL_SCENARIOS}"
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

echo "📍 Using checkpoints:"
is_enabled "$RUN_ZERO_SHOT" && echo "  Zero-shot:       $ZERO_SHOT_CKPT (PLUTO, no fine-tuning)"
is_enabled "$RUN_RULE_BASED" && echo "  Rule-based:      $RULE_BASED_CKPT (PLUTO + rule-based curriculum LoRA)"
is_enabled "$RUN_LOSS_BASED" && echo "  Loss-based:      $LOSS_BASED_CKPT (PLUTO + loss-ranked curriculum LoRA)"
is_enabled "$RUN_UNIFORM" && echo "  Uniform:         $UNIFORM_CKPT (PLUTO + uniform-principle curriculum LoRA)"
is_enabled "$RUN_RANDOM_BUCKET" && echo "  RandomBucket-FT: $RANDOM_BUCKET_CKPT (PLUTO + random-bucket curriculum LoRA)"
is_enabled "$RUN_LLM_CURRICULUM" && echo "  LLM curriculum:  $CURRICULUM_CKPT (PLUTO + LLM-based curriculum LoRA)"
echo ""
echo "📍 Using interPlan scenario filter: $INTERPLAN_FILTER"
echo ""

START_TIME=$(date +%s)

################################################################################
# Test: InterPlan Benchmark
################################################################################
echo ""
echo "=============================================="
echo "Testing InterPlan Benchmark ($INTERPLAN_FILTER)"
echo "=============================================="

# Create experiment name suffix based on filter
if [ "$INTERPLAN_FILTER" = "interplan10" ]; then
    EXP_SUFFIX="interplan10"
else
    EXP_SUFFIX="benchmark_scenarios"
fi
run_enabled_interplan_models "$EXP_SUFFIX"

################################################################################
# Summary and Analysis
################################################################################

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=============================================="
echo "✅ Quick test (interPlan $INTERPLAN_FILTER) complete!"
echo "=============================================="
echo "Time taken: ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results are in: ${NUPLAN_EXP_ROOT}/exp/quick_test_interplan_*_${EXP_SUFFIX}"
echo ""
echo "Analyzing results..."
python ${REPO_ROOT}/scripts/analysis/analyze_quick_test.py

echo ""
echo "=============================================="
echo "Next steps:"
echo "  1. Check if metrics are present"
echo "  2. Compare results with other test datasets"
echo "  3. Verify all scenarios were processed correctly"
echo "=============================================="
