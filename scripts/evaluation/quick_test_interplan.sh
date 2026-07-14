#!/bin/bash
################################################################################
# Quick Test: InterPlan Benchmark
# Uses interPlan's official scenario filters (interplan10 or benchmark_scenarios)
# Validates that metrics are collected correctly
#
# This script runs interPlan benchmark with official filters
# for enabled methods (zero-shot, rule-based, loss-based, uniform, random-bucket, LLM curriculum, MPOC)
#
# Usage:
#   ./quick_test_interplan.sh [interplan10|benchmark_scenarios]
#   Default: interplan10 (80 scenarios, 10 per type)
################################################################################

set -e

ACTIVE_QUICK_TEST_LOCK=""

cleanup_quick_test_lock() {
    if [ -n "$ACTIVE_QUICK_TEST_LOCK" ]; then
        rm -rf "$ACTIVE_QUICK_TEST_LOCK"
    fi
}

trap cleanup_quick_test_lock EXIT INT TERM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
NUPLAN_DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-${WORKSPACE_ROOT}/nuplan-devkit}"
INTERPLAN_ROOT="${WORKSPACE_ROOT}/interPlan"
cd "$REPO_ROOT"

apply_cli_overrides() {
    local positional_filter=""
    for arg in "$@"; do
        case "$arg" in
            INTERPLAN_FILTER=*|EXPERIMENT_SUFFIX=*|COLLECT_TEST=*|\
            RUN_ZERO_SHOT=*|RUN_RULE_BASED=*|RUN_LOSS_BASED=*|RUN_UNIFORM=*|RUN_RANDOM_BUCKET=*|RUN_LLM_CURRICULUM=*|RUN_MPOC=*|\
            LLM_CURRICULUM_VERSION=*|LLM_CURRICULUM_SLUG=*|LLM_CURRICULUM_EXP=*|\
            UNIFORM_CURRICULUM_VERSION=*|UNIFORM_CURRICULUM_SLUG=*|UNIFORM_CURRICULUM_EXP=*|\
            RULE_CURRICULUM_VERSION=*|RULE_CURRICULUM_SLUG=*|RULE_CURRICULUM_EXP=*|\
            LOSS_CURRICULUM_VERSION=*|LOSS_CURRICULUM_SLUG=*|LOSS_CURRICULUM_EXP=*|\
            RANDOM_BUCKET_CURRICULUM_VERSION=*|RANDOM_BUCKET_CURRICULUM_SLUG=*|RANDOM_BUCKET_CURRICULUM_EXP=*|\
            MPOC_CURRICULUM_VERSION=*|MPOC_CURRICULUM_SLUG=*|MPOC_CURRICULUM_EXP=*|\
            PERCENTILE_EHU_FINAL_PHASE=*)
                export "$arg"
                ;;
            interplan10|benchmark_scenarios)
                if [ -n "$positional_filter" ]; then
                    echo "Error: multiple interPlan filters provided: $positional_filter and $arg"
                    exit 1
                fi
                positional_filter="$arg"
                ;;
            *)
                echo "Error: unsupported argument: $arg"
                echo "Use [interplan10|benchmark_scenarios] and/or supported KEY=value overrides."
                exit 1
                ;;
        esac
    done

    if [ -n "$positional_filter" ]; then
        export INTERPLAN_FILTER="$positional_filter"
    fi
}

apply_cli_overrides "$@"

# Configuration: InterPlan scenario filter
# - interplan10: Official benchmark with 80 scenarios (10 per type)
# - benchmark_scenarios: All 335 scenarios
INTERPLAN_FILTER=${INTERPLAN_FILTER:-benchmark_scenarios} # interplan10 or benchmark_scenarios

# Model selection flags. Set any flag to false/0/no to skip that model.
RUN_ZERO_SHOT=${RUN_ZERO_SHOT:-true}
RUN_RULE_BASED=${RUN_RULE_BASED:-true}
RUN_LOSS_BASED=${RUN_LOSS_BASED:-false}
RUN_UNIFORM=${RUN_UNIFORM:-true}
RUN_RANDOM_BUCKET=${RUN_RANDOM_BUCKET:-false}
RUN_LLM_CURRICULUM=${RUN_LLM_CURRICULUM:-true}
RUN_MPOC=${RUN_MPOC:-true}

LLM_CURRICULUM_VERSION=${LLM_CURRICULUM_VERSION:-v4.3.12}
LLM_CURRICULUM_SLUG=${LLM_CURRICULUM_SLUG:-curriculum_llm_guided_${LLM_CURRICULUM_VERSION}}
LLM_CURRICULUM_EXP=${LLM_CURRICULUM_EXP:-curriculum_lora_llm_guided_${LLM_CURRICULUM_VERSION}_stage3_high}

UNIFORM_CURRICULUM_VERSION=${UNIFORM_CURRICULUM_VERSION:-v2.3.9}
UNIFORM_CURRICULUM_SLUG=${UNIFORM_CURRICULUM_SLUG:-curriculum_uniform_${UNIFORM_CURRICULUM_VERSION}}
UNIFORM_CURRICULUM_EXP=${UNIFORM_CURRICULUM_EXP:-curriculum_lora_uniform_${UNIFORM_CURRICULUM_VERSION}_stage3_uniform}

PERCENTILE_EHU_FINAL_PHASE=${PERCENTILE_EHU_FINAL_PHASE:-phaseC_uniform_consolidation}

RULE_CURRICULUM_VERSION=${RULE_CURRICULUM_VERSION:-}
if [ -n "$RULE_CURRICULUM_VERSION" ]; then
    RULE_CURRICULUM_SLUG=${RULE_CURRICULUM_SLUG:-curriculum_rule_percentile_ehu_${RULE_CURRICULUM_VERSION}}
    RULE_CURRICULUM_EXP=${RULE_CURRICULUM_EXP:-curriculum_lora_rule_percentile_ehu_${RULE_CURRICULUM_VERSION}_${PERCENTILE_EHU_FINAL_PHASE}}
else
    RULE_CURRICULUM_SLUG=${RULE_CURRICULUM_SLUG:-rulebased}
    RULE_CURRICULUM_EXP=${RULE_CURRICULUM_EXP:-curriculum_lora_rulebased_stage3_high}
fi

LOSS_CURRICULUM_VERSION=${LOSS_CURRICULUM_VERSION:-}
if [ -n "$LOSS_CURRICULUM_VERSION" ]; then
    LOSS_CURRICULUM_SLUG=${LOSS_CURRICULUM_SLUG:-curriculum_loss_percentile_ehu_${LOSS_CURRICULUM_VERSION}}
    LOSS_CURRICULUM_EXP=${LOSS_CURRICULUM_EXP:-curriculum_lora_loss_percentile_ehu_${LOSS_CURRICULUM_VERSION}_${PERCENTILE_EHU_FINAL_PHASE}}
else
    LOSS_CURRICULUM_SLUG=${LOSS_CURRICULUM_SLUG:-lossbased}
    LOSS_CURRICULUM_EXP=${LOSS_CURRICULUM_EXP:-curriculum_lora_lossrank_stage3_high}
fi

RANDOM_BUCKET_CURRICULUM_VERSION=${RANDOM_BUCKET_CURRICULUM_VERSION:-}
if [ -n "$RANDOM_BUCKET_CURRICULUM_VERSION" ]; then
    RANDOM_BUCKET_CURRICULUM_SLUG=${RANDOM_BUCKET_CURRICULUM_SLUG:-curriculum_randombucket_percentile_ehu_${RANDOM_BUCKET_CURRICULUM_VERSION}}
    RANDOM_BUCKET_CURRICULUM_EXP=${RANDOM_BUCKET_CURRICULUM_EXP:-curriculum_lora_random_percentile_ehu_${RANDOM_BUCKET_CURRICULUM_VERSION}_${PERCENTILE_EHU_FINAL_PHASE}}
else
    RANDOM_BUCKET_CURRICULUM_SLUG=${RANDOM_BUCKET_CURRICULUM_SLUG:-curriculum_randombucket}
    RANDOM_BUCKET_CURRICULUM_EXP=${RANDOM_BUCKET_CURRICULUM_EXP:-curriculum_lora_randombucket_stage3_high}
fi

MPOC_CURRICULUM_VERSION=${MPOC_CURRICULUM_VERSION:-}
if [ -n "$MPOC_CURRICULUM_VERSION" ]; then
    MPOC_CURRICULUM_SLUG=${MPOC_CURRICULUM_SLUG:-curriculum_mpoc_percentile_ehu_${MPOC_CURRICULUM_VERSION}}
    MPOC_CURRICULUM_EXP=${MPOC_CURRICULUM_EXP:-curriculum_lora_mpoc_percentile_ehu_${MPOC_CURRICULUM_VERSION}_${PERCENTILE_EHU_FINAL_PHASE}}
else
    MPOC_CURRICULUM_SLUG=${MPOC_CURRICULUM_SLUG:-curriculum_mpoc}
    MPOC_CURRICULUM_EXP=${MPOC_CURRICULUM_EXP:-curriculum_lora_mpoc_stage3_high}
fi

if [ "$INTERPLAN_FILTER" != "interplan10" ] && [ "$INTERPLAN_FILTER" != "benchmark_scenarios" ]; then
    echo "❌ Error: Invalid filter. Use 'interplan10' or 'benchmark_scenarios'"
    exit 1
fi

# InterPlan paths
INTERPLAN_SCRIPT="${INTERPLAN_ROOT}/interplan/planning/script/run_simulation.py"
WANDB_SHIM_ROOT="${SCRIPT_DIR}/wandb_disabled_shim"

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
    local eval_pythonpath="${PYTHONPATH:-}"
    local simulation_log_overrides=()
    is_enabled "${DISABLE_SIMULATION_LOG:-false}" && simulation_log_overrides=(callback=no_simulation_log)

    if [ "${PLUTO_EVAL_ALLOW_WANDB:-0}" != "1" ]; then
        eval_pythonpath="${WANDB_SHIM_ROOT}:${eval_pythonpath}"
    fi
    
    echo "   Running interPlan simulation with checkpoint: $ckpt"
    
    # Use interPlan's official format (following sim_pdm_closed.sh example)
    # Add pluto config directory to searchpath so pluto_planner can be found
    # Override scenario_builder data_root to point to correct location
    # interPlan's default interplan.yaml points to trainval which may not exist
    # We need to find where the actual data is located
    WANDB_DISABLED="${WANDB_DISABLED:-true}" \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}" \
    PYTHONPATH="$eval_pythonpath" \
    python - "$INTERPLAN_SCRIPT" \
        +simulation=default_interplan_benchmark \
        scenario_filter=$filter \
        scenario_builder.data_root='${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1_test/data/cache/test' \
        scenario_builder.sensor_root='${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1_test/sensor_blobs' \
        planner=pluto_planner \
        +planner.pluto_planner.planner_ckpt="$ckpt" \
        experiment_name="$experiment" \
        "${simulation_log_overrides[@]}" \
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
]" <<'PY'
import os
import runpy
import sys

script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

runpy.run_path(script, run_name="__main__")
PY
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

run_model_interplan_simulation() {
    local label=$1
    local slug=$2
    local ckpt=$3
    local experiment_suffix=$4
    local experiment="quick_test_interplan_${slug}_${experiment_suffix}"
    local lock_root="${NUPLAN_EXP_ROOT}/exp/.quick_test_locks"
    local lock_dir="${lock_root}/${experiment}.lock"

    echo ""
    echo "Running ${label} on interPlan (${INTERPLAN_FILTER})..."
    mkdir -p "$lock_root"
    if [ -f "${lock_dir}/pid" ]; then
        local existing_pid
        existing_pid="$(cat "${lock_dir}/pid" 2>/dev/null || true)"
        local existing_cmd
        existing_cmd=""
        if [ -n "$existing_pid" ] && kill -0 "$existing_pid" 2>/dev/null; then
            existing_cmd="$(ps -p "$existing_pid" -o args= 2>/dev/null || true)"
        fi
        if [ -z "$existing_pid" ] || ! kill -0 "$existing_pid" 2>/dev/null; then
            echo "Removing stale quick-test lock for ${experiment} (pid ${existing_pid:-unknown} is not running)."
            rm -rf "$lock_dir"
        elif [[ "$existing_cmd" != *"quick_test_interplan"* && "$existing_cmd" != *"run_simulation.py"* ]]; then
            echo "Removing stale quick-test lock for ${experiment} (pid ${existing_pid} is not a quick-test process)."
            rm -rf "$lock_dir"
        fi
    fi
    if ! mkdir "$lock_dir" 2>/dev/null; then
        echo "Error: ${experiment} appears to be running already."
        echo "   Lock directory: ${lock_dir}"
        echo "   Running the same quick-test experiment concurrently can corrupt metrics."
        exit 1
    fi
    echo "$$" > "${lock_dir}/pid"
    ACTIVE_QUICK_TEST_LOCK="$lock_dir"

    set +e
    run_interplan_simulation "$INTERPLAN_FILTER" "$ckpt" "$experiment"
    local simulation_status=$?
    set -e
    rm -rf "$lock_dir"
    ACTIVE_QUICK_TEST_LOCK=""
    if [ "$simulation_status" -ne 0 ]; then
        return "$simulation_status"
    fi

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
    is_enabled "$RUN_RULE_BASED" && run_model_interplan_simulation "Rule-based${RULE_CURRICULUM_VERSION:+ (${RULE_CURRICULUM_VERSION})}" "$RULE_CURRICULUM_SLUG" "$RULE_BASED_CKPT" "$experiment_suffix"
    is_enabled "$RUN_LOSS_BASED" && run_model_interplan_simulation "Loss-based${LOSS_CURRICULUM_VERSION:+ (${LOSS_CURRICULUM_VERSION})}" "$LOSS_CURRICULUM_SLUG" "$LOSS_BASED_CKPT" "$experiment_suffix"
    is_enabled "$RUN_UNIFORM" && run_model_interplan_simulation "Uniform FT (${UNIFORM_CURRICULUM_VERSION})" "$UNIFORM_CURRICULUM_SLUG" "$UNIFORM_CKPT" "$experiment_suffix"
    is_enabled "$RUN_RANDOM_BUCKET" && run_model_interplan_simulation "RandomBucket-FT${RANDOM_BUCKET_CURRICULUM_VERSION:+ (${RANDOM_BUCKET_CURRICULUM_VERSION})}" "$RANDOM_BUCKET_CURRICULUM_SLUG" "$RANDOM_BUCKET_CKPT" "$experiment_suffix"
    is_enabled "$RUN_LLM_CURRICULUM" && run_model_interplan_simulation "LLM-guided curriculum (${LLM_CURRICULUM_VERSION})" "$LLM_CURRICULUM_SLUG" "$CURRICULUM_CKPT" "$experiment_suffix"
    is_enabled "$RUN_MPOC" && run_model_interplan_simulation "MPOC curriculum${MPOC_CURRICULUM_VERSION:+ (${MPOC_CURRICULUM_VERSION})}" "$MPOC_CURRICULUM_SLUG" "$MPOC_CKPT" "$experiment_suffix"

    # Disabled trailing methods must not turn a successful selected model into
    # a false benchmark failure under the parent suite.
    return 0
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

is_enabled "$RUN_RULE_BASED" && find_lora_checkpoint RULE_BASED_CKPT "Rule-based" "$RULE_CURRICULUM_EXP"
is_enabled "$RUN_LOSS_BASED" && find_lora_checkpoint LOSS_BASED_CKPT "Loss-based" "$LOSS_CURRICULUM_EXP"
is_enabled "$RUN_UNIFORM" && find_lora_checkpoint UNIFORM_CKPT "Uniform FT" "$UNIFORM_CURRICULUM_EXP"
is_enabled "$RUN_RANDOM_BUCKET" && find_lora_checkpoint RANDOM_BUCKET_CKPT "RandomBucket-FT" "$RANDOM_BUCKET_CURRICULUM_EXP"
is_enabled "$RUN_LLM_CURRICULUM" && find_lora_checkpoint CURRICULUM_CKPT "LLM-guided curriculum" "$LLM_CURRICULUM_EXP"
is_enabled "$RUN_MPOC" && find_lora_checkpoint MPOC_CKPT "MPOC curriculum" "$MPOC_CURRICULUM_EXP"

echo "📍 Using checkpoints:"
is_enabled "$RUN_ZERO_SHOT" && echo "  Zero-shot:       $ZERO_SHOT_CKPT (PLUTO, no fine-tuning)"
is_enabled "$RUN_RULE_BASED" && echo "  Rule-based:      $RULE_BASED_CKPT (PLUTO + rule-based curriculum LoRA, slug=${RULE_CURRICULUM_SLUG}, exp=${RULE_CURRICULUM_EXP})"
is_enabled "$RUN_LOSS_BASED" && echo "  Loss-based:      $LOSS_BASED_CKPT (PLUTO + loss-ranked curriculum LoRA, slug=${LOSS_CURRICULUM_SLUG}, exp=${LOSS_CURRICULUM_EXP})"
is_enabled "$RUN_UNIFORM" && echo "  Uniform FT:      $UNIFORM_CKPT (PLUTO + ${UNIFORM_CURRICULUM_VERSION} uniform FT LoRA, slug=${UNIFORM_CURRICULUM_SLUG})"
is_enabled "$RUN_RANDOM_BUCKET" && echo "  RandomBucket-FT: $RANDOM_BUCKET_CKPT (PLUTO + random-bucket curriculum LoRA, slug=${RANDOM_BUCKET_CURRICULUM_SLUG}, exp=${RANDOM_BUCKET_CURRICULUM_EXP})"
is_enabled "$RUN_LLM_CURRICULUM" && echo "  LLM-guided:      $CURRICULUM_CKPT (PLUTO + ${LLM_CURRICULUM_VERSION} curriculum LoRA, slug=${LLM_CURRICULUM_SLUG})"
is_enabled "$RUN_MPOC" && echo "  MPOC curriculum: $MPOC_CKPT (PLUTO + MPOC curriculum LoRA, slug=${MPOC_CURRICULUM_SLUG}, exp=${MPOC_CURRICULUM_EXP})"
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
if [ -n "${EXPERIMENT_SUFFIX:-}" ]; then
    EXP_SUFFIX="$EXPERIMENT_SUFFIX"
elif [ "$INTERPLAN_FILTER" = "interplan10" ]; then
    EXP_SUFFIX="interplan10"
else
    EXP_SUFFIX="benchmark_scenarios"
fi

if [ -z "${COLLECT_TEST:-}" ]; then
    if [ "$EXP_SUFFIX" = "interplan10" ]; then
        COLLECT_TEST="interplan10"
    else
        COLLECT_TEST="interplan-benchmark"
    fi
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
echo "Collecting result summary..."

COLLECT_METHOD_KEYS=()
is_enabled "$RUN_ZERO_SHOT" && COLLECT_METHOD_KEYS+=("zeroshot")
is_enabled "$RUN_RULE_BASED" && COLLECT_METHOD_KEYS+=("$RULE_CURRICULUM_SLUG")
is_enabled "$RUN_LOSS_BASED" && COLLECT_METHOD_KEYS+=("$LOSS_CURRICULUM_SLUG")
is_enabled "$RUN_UNIFORM" && COLLECT_METHOD_KEYS+=("$UNIFORM_CURRICULUM_SLUG")
is_enabled "$RUN_RANDOM_BUCKET" && COLLECT_METHOD_KEYS+=("$RANDOM_BUCKET_CURRICULUM_SLUG")
is_enabled "$RUN_LLM_CURRICULUM" && COLLECT_METHOD_KEYS+=("$LLM_CURRICULUM_SLUG")
is_enabled "$RUN_MPOC" && COLLECT_METHOD_KEYS+=("$MPOC_CURRICULUM_SLUG")
COLLECT_METHODS=$(IFS=,; echo "${COLLECT_METHOD_KEYS[*]}")

python ${REPO_ROOT}/scripts/evaluation/collect_quick_test_results.py \
    --tests "$COLLECT_TEST" \
    --methods "$COLLECT_METHODS" \
    --detail || echo "Could not collect interPlan (${INTERPLAN_FILTER}) summary"

echo ""
echo "=============================================="
echo "Next steps:"
echo "  1. Check if metrics are present"
echo "  2. Compare results with other test datasets"
echo "  3. Verify all scenarios were processed correctly"
echo "=============================================="
