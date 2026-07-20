#!/bin/bash
################################################################################
# Quick Test: Configurable scenarios per group
# Validates that metrics are collected correctly
# Estimated time: 3-4 hours
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
WANDB_SHIM_ROOT="${SCRIPT_DIR}/wandb_disabled_shim"
cd "$REPO_ROOT"

configure_eval_import_environment() {
    if [ "${PLUTO_EVAL_ALLOW_WANDB:-0}" != "1" ] && [ -f "${WANDB_SHIM_ROOT}/wandb.py" ]; then
        case ":${PYTHONPATH:-}:" in
            *":${WANDB_SHIM_ROOT}:"*) ;;
            *) export PYTHONPATH="${WANDB_SHIM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
        esac
    fi

    export WANDB_DISABLED="${WANDB_DISABLED:-true}"
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
}

apply_cli_overrides() {
    for arg in "$@"; do
        case "$arg" in
            FILTER_NAME=*|EXPERIMENT_SUFFIX=*|TEST_LABEL=*|SCENARIO_BUILDER=*|COLLECT_TEST=*|\
            RUN_ZERO_SHOT=*|RUN_RULE_BASED=*|RUN_LOSS_BASED=*|RUN_UNIFORM=*|RUN_RANDOM_BUCKET=*|RUN_LLM_CURRICULUM=*|RUN_MPOC=*|\
            SCENARIOS_PER_STAGE=*|BATCH_SIZE=*|SIMULATION_TYPE=*|SIMULATION_VERBOSE=*|ENABLE_PROGRESS_BAR=*|\
            SIMULATION_WORKER=*|SIMULATION_WORKER_THREADS=*|SIMULATION_WORKER_MAX_WORKERS=*|SIMULATION_NUM_GPUS=*|SIMULATION_NUM_CPUS=*|SIMULATION_RAY_LOG_TO_DRIVER=*|\
            PLUTO_EVAL_ALLOW_WANDB=*|WANDB_DISABLED=*|PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=*|\
            LLM_CURRICULUM_VERSION=*|LLM_CURRICULUM_SLUG=*|LLM_CURRICULUM_EXP=*|\
            UNIFORM_CURRICULUM_VERSION=*|UNIFORM_CURRICULUM_SLUG=*|UNIFORM_CURRICULUM_EXP=*|\
            RULE_CURRICULUM_VERSION=*|RULE_CURRICULUM_SLUG=*|RULE_CURRICULUM_EXP=*|\
            LOSS_CURRICULUM_VERSION=*|LOSS_CURRICULUM_SLUG=*|LOSS_CURRICULUM_EXP=*|\
            RANDOM_BUCKET_CURRICULUM_VERSION=*|RANDOM_BUCKET_CURRICULUM_SLUG=*|RANDOM_BUCKET_CURRICULUM_EXP=*|\
            MPOC_CURRICULUM_VERSION=*|MPOC_CURRICULUM_SLUG=*|MPOC_CURRICULUM_EXP=*|\
            PERCENTILE_EHU_FINAL_PHASE=*)
                export "$arg"
                ;;
            *)
                echo "Error: unsupported argument: $arg"
                echo "Use KEY=value before the command, or one of the supported KEY=value overrides after the script."
                exit 1
                ;;
        esac
    done
}

apply_cli_overrides "$@"
configure_eval_import_environment

# Configuration: Number of scenarios to evaluate per stage.
# "auto" uses every token explicitly listed in the selected filter.
# This ensures all enabled methods use the same scenarios.
# Note: Sequential worker is used to minimize memory usage. 
# WARNING: If you get OOM (Out Of Memory) errors, enable batch processing (see BATCH_SIZE below).
# The simulation framework builds all simulation objects upfront, so large scenario counts can cause OOM.
SCENARIOS_PER_STAGE=${SCENARIOS_PER_STAGE:-auto}
FILTER_NAME=${FILTER_NAME:-val14_benchmark}
EXPERIMENT_SUFFIX=${EXPERIMENT_SUFFIX:-val14_benchmark}
TEST_LABEL=${TEST_LABEL:-Val14 Benchmark}
SCENARIO_BUILDER=${SCENARIO_BUILDER:-nuplan_v1_1_val}
COLLECT_TEST=${COLLECT_TEST:-val14-benchmark}

# Model selection flags. Set any flag to false/0/no to skip that model.
RUN_ZERO_SHOT=${RUN_ZERO_SHOT:-false}
RUN_RULE_BASED=${RUN_RULE_BASED:-false}
RUN_LOSS_BASED=${RUN_LOSS_BASED:-false}
RUN_UNIFORM=${RUN_UNIFORM:-false}
RUN_RANDOM_BUCKET=${RUN_RANDOM_BUCKET:-false}
RUN_LLM_CURRICULUM=${RUN_LLM_CURRICULUM:-false}
RUN_MPOC=${RUN_MPOC:-false}

LLM_CURRICULUM_VERSION=${LLM_CURRICULUM_VERSION:-v4.3.12}
LLM_CURRICULUM_SLUG=${LLM_CURRICULUM_SLUG:-curriculum_llm_guided_${LLM_CURRICULUM_VERSION}}
LLM_CURRICULUM_EXP=${LLM_CURRICULUM_EXP:-curriculum_lora_llm_guided_${LLM_CURRICULUM_VERSION}_stage3_high}

UNIFORM_CURRICULUM_VERSION=${UNIFORM_CURRICULUM_VERSION:-v2.3.9}
UNIFORM_CURRICULUM_SLUG=${UNIFORM_CURRICULUM_SLUG:-curriculum_uniform_${UNIFORM_CURRICULUM_VERSION}}
UNIFORM_CURRICULUM_EXP=${UNIFORM_CURRICULUM_EXP:-curriculum_lora_uniform_${UNIFORM_CURRICULUM_VERSION}_stage3_uniform}

PERCENTILE_EHU_FINAL_PHASE=${PERCENTILE_EHU_FINAL_PHASE:-phaseC_hard_replay}

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

# Batch size for processing scenarios (to avoid OOM)
# If set to a positive number, scenarios will be automatically split into batches and processed sequentially.
# Example: BATCH_SIZE=200 will process scenarios in batches of 200.
# Set to empty or 0 to disable batching (not recommended for large scenario counts).
# Recommended: 150-300 depending on available memory
BATCH_SIZE=${BATCH_SIZE:-50}

# Simulation type: nonreactive preserves the historical Val14 quick-test path.
# Set to "reactive" for closed_loop_reactive_agents.
SIMULATION_TYPE=${SIMULATION_TYPE:-nonreactive}
SIMULATION_VERBOSE=${SIMULATION_VERBOSE:-true}
ENABLE_PROGRESS_BAR=${ENABLE_PROGRESS_BAR:-true}

# InterPlan-style scenario scheduling. With num_gpus=0, Ray hides CUDA from
# workers and PLUTO falls back to CPU, allowing multiple scenario workers
# without several processes competing for the same GPU.
SIMULATION_WORKER=${SIMULATION_WORKER:-ray_distributed}
SIMULATION_WORKER_THREADS=${SIMULATION_WORKER_THREADS:-}
SIMULATION_WORKER_MAX_WORKERS=${SIMULATION_WORKER_MAX_WORKERS:-}
SIMULATION_NUM_GPUS=${SIMULATION_NUM_GPUS:-0}
SIMULATION_NUM_CPUS=${SIMULATION_NUM_CPUS:-1}
SIMULATION_RAY_LOG_TO_DRIVER=${SIMULATION_RAY_LOG_TO_DRIVER:-true}

WORKER_OVERRIDES=()

build_worker_overrides() {
    WORKER_OVERRIDES=(
        "worker=${SIMULATION_WORKER}"
        "number_of_gpus_allocated_per_simulation=${SIMULATION_NUM_GPUS}"
        "number_of_cpus_allocated_per_simulation=${SIMULATION_NUM_CPUS}"
    )
    if [ "$SIMULATION_WORKER" = "ray_distributed" ]; then
        [ -n "$SIMULATION_WORKER_THREADS" ] && WORKER_OVERRIDES+=("worker.threads_per_node=${SIMULATION_WORKER_THREADS}")
        [ -n "$SIMULATION_RAY_LOG_TO_DRIVER" ] && WORKER_OVERRIDES+=("worker.log_to_driver=${SIMULATION_RAY_LOG_TO_DRIVER}")
    elif [ "$SIMULATION_WORKER" = "single_machine_thread_pool" ]; then
        [ -n "$SIMULATION_WORKER_MAX_WORKERS" ] && WORKER_OVERRIDES+=("worker.max_workers=${SIMULATION_WORKER_MAX_WORKERS}")
    fi
}

batched_worker_args() {
    printf '%s\n' \
        --worker "$SIMULATION_WORKER" \
        --num-gpus "$SIMULATION_NUM_GPUS" \
        --num-cpus "$SIMULATION_NUM_CPUS"
    if [ "$SIMULATION_WORKER" = "ray_distributed" ]; then
        [ -n "$SIMULATION_WORKER_THREADS" ] && printf '%s\n' --worker-threads "$SIMULATION_WORKER_THREADS"
        [ -n "$SIMULATION_RAY_LOG_TO_DRIVER" ] && printf '%s\n' --ray-log-to-driver "$SIMULATION_RAY_LOG_TO_DRIVER"
    elif [ "$SIMULATION_WORKER" = "single_machine_thread_pool" ]; then
        [ -n "$SIMULATION_WORKER_MAX_WORKERS" ] && printf '%s\n' --worker-max-workers "$SIMULATION_WORKER_MAX_WORKERS"
    fi
}

# Helper function to run simulation (with automatic batching if enabled)
run_simulation() {
    local filter=$1
    local ckpt=$2
    local experiment=$3
    local scenario_builder=${4:-""}
    local simulation_log_args=()
    is_enabled "${DISABLE_SIMULATION_LOG:-false}" && simulation_log_args=(--disable-simulation-log)
    local simulation_log_overrides=()
    is_enabled "${DISABLE_SIMULATION_LOG:-false}" && simulation_log_overrides=(callback=no_simulation_log)

    if [ "$SIMULATION_TYPE" = "reactive" ]; then
        local simulation_config="closed_loop_reactive_agents"
        local observation_config="idm_agents_observation"
    else
        local simulation_config="closed_loop_nonreactive_agents"
        local observation_config="box_observation"
    fi
    
    if [ -n "$BATCH_SIZE" ] && [ "$BATCH_SIZE" -gt 0 ] && [ "$SIMULATION_TYPE" != "reactive" ]; then
        local builder_arg=""
        [ -n "$scenario_builder" ] && builder_arg="--scenario-builder $scenario_builder"
        local worker_args=()
        mapfile -t worker_args < <(batched_worker_args)
        
        python ${REPO_ROOT}/scripts/evaluation/run_simulation_batched.py \
            --filter "$filter" \
            --ckpt "$ckpt" \
            --experiment "$experiment" \
            --batch-size $BATCH_SIZE \
            --limit $SCENARIOS_PER_STAGE \
            --simulation-verbose "$SIMULATION_VERBOSE" \
            "${simulation_log_args[@]}" \
            "${worker_args[@]}" \
            $builder_arg
    else
        if [ -n "$BATCH_SIZE" ] && [ "$BATCH_SIZE" -gt 0 ] && [ "$SIMULATION_TYPE" = "reactive" ]; then
            echo "⚠️  Warning: Batching is not yet supported for reactive agents."
            echo "   Running without batching..."
        fi

        local builder_arg=""
        [ -n "$scenario_builder" ] && builder_arg="scenario_builder=$scenario_builder"
        build_worker_overrides
        
        python -X faulthandler ${REPO_ROOT}/run_simulation.py \
            +simulation=$simulation_config \
            observation=$observation_config \
            ego_controller=two_stage_controller \
            planner=pluto_planner \
            +planner.pluto_planner.planner_ckpt="$ckpt" \
            scenario_filter="$filter" \
            scenario_filter.limit_total_scenarios=$SCENARIOS_PER_STAGE \
            verbose="$SIMULATION_VERBOSE" \
            enable_simulation_progress_bar="$ENABLE_PROGRESS_BAR" \
            $builder_arg \
            experiment="$experiment" \
            "${simulation_log_overrides[@]}" \
            "${WORKER_OVERRIDES[@]}"
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

    # The canonical paper runner resolves and validates the checkpoint once,
    # then passes it to this single-model leaf adapter.  Legacy callers may
    # continue to rely on experiment-name discovery.
    if [ -n "${PLUTO_EVAL_CHECKPOINT:-}" ]; then
        if [ ! -f "$PLUTO_EVAL_CHECKPOINT" ]; then
            echo "Error: explicit ${label} checkpoint not found: $PLUTO_EVAL_CHECKPOINT" >&2
            exit 1
        fi
        printf -v "$result_var" '%s' "$PLUTO_EVAL_CHECKPOINT"
        return 0
    fi

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
    local lock_root="${NUPLAN_EXP_ROOT}/exp/.quick_test_locks"
    local lock_dir="${lock_root}/${experiment}.lock"

    echo ""
    echo "Running ${label} on ${filter}..."
    mkdir -p "$lock_root"
    # Preserve the NFS write group even when a quick test is launched through
    # a root docker exec.  A later non-root run must be able to remove its
    # stale lock after a crash or container recreation.
    chmod 2775 "$lock_root" 2>/dev/null || true
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
        elif [[ "$existing_cmd" != *"quick_test_val14"* && "$existing_cmd" != *"run_simulation.py"* ]]; then
            echo "Removing stale quick-test lock for ${experiment} (pid ${existing_pid} is not a quick-test process)."
            rm -rf "$lock_dir"
        fi
    fi
    if ! (umask 0002; mkdir "$lock_dir") 2>/dev/null; then
        echo "Error: ${experiment} appears to be running already."
        echo "   Lock directory: ${lock_dir}"
        echo "   Running the same quick-test experiment concurrently can corrupt metrics."
        exit 1
    fi
    chmod 2775 "$lock_dir" 2>/dev/null || true
    (umask 0002; echo "$$" > "${lock_dir}/pid")
    ACTIVE_QUICK_TEST_LOCK="$lock_dir"

    set +e
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
            worker.max_workers="${SIMULATION_WORKER_MAX_WORKERS:-1}" \
            number_of_gpus_allocated_per_simulation="$SIMULATION_NUM_GPUS" \
            number_of_cpus_allocated_per_simulation="$SIMULATION_NUM_CPUS"
    else
        run_simulation "$filter" "$ckpt" "$experiment" "$scenario_builder"
    fi
    local simulation_status=$?
    set -e
    rm -rf "$lock_dir"
    ACTIVE_QUICK_TEST_LOCK=""
    if [ "$simulation_status" -ne 0 ]; then
        return "$simulation_status"
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
    is_enabled "$RUN_RULE_BASED" && run_model_simulation "Rule-based${RULE_CURRICULUM_VERSION:+ (${RULE_CURRICULUM_VERSION})}" "$RULE_CURRICULUM_SLUG" "$RULE_BASED_CKPT" "$filter" "$experiment_suffix" "$scenario_builder" "$run_mode"
    is_enabled "$RUN_LOSS_BASED" && run_model_simulation "Loss-based${LOSS_CURRICULUM_VERSION:+ (${LOSS_CURRICULUM_VERSION})}" "$LOSS_CURRICULUM_SLUG" "$LOSS_BASED_CKPT" "$filter" "$experiment_suffix" "$scenario_builder" "$run_mode"
    is_enabled "$RUN_UNIFORM" && run_model_simulation "Uniform FT (${UNIFORM_CURRICULUM_VERSION})" "$UNIFORM_CURRICULUM_SLUG" "$UNIFORM_CKPT" "$filter" "$experiment_suffix" "$scenario_builder" "$run_mode"
    is_enabled "$RUN_RANDOM_BUCKET" && run_model_simulation "RandomBucket-FT${RANDOM_BUCKET_CURRICULUM_VERSION:+ (${RANDOM_BUCKET_CURRICULUM_VERSION})}" "$RANDOM_BUCKET_CURRICULUM_SLUG" "$RANDOM_BUCKET_CKPT" "$filter" "$experiment_suffix" "$scenario_builder" "$run_mode"
    is_enabled "$RUN_LLM_CURRICULUM" && run_model_simulation "LLM-guided curriculum (${LLM_CURRICULUM_VERSION})" "$LLM_CURRICULUM_SLUG" "$CURRICULUM_CKPT" "$filter" "$experiment_suffix" "$scenario_builder" "$run_mode"
    is_enabled "$RUN_MPOC" && run_model_simulation "MPOC curriculum${MPOC_CURRICULUM_VERSION:+ (${MPOC_CURRICULUM_VERSION})}" "$MPOC_CURRICULUM_SLUG" "$MPOC_CKPT" "$filter" "$experiment_suffix" "$scenario_builder" "$run_mode"

    # Disabled trailing methods must not turn a successful selected model into
    # a false benchmark failure under the parent suite.
    return 0
}

# Set up Python/runtime paths. Supports conda, .venv, or an already-active env.
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/env_bootstrap.sh"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/python_runtime.sh"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/evaluation/local_benchmark_data.sh"
use_local_benchmark_data val14

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
echo "Simulation type: ${SIMULATION_TYPE}"
echo "Simulation verbose: ${SIMULATION_VERBOSE}"
echo "Progress bar: ${ENABLE_PROGRESS_BAR}"
echo "Worker: ${SIMULATION_WORKER} (num_gpus=${SIMULATION_NUM_GPUS}, num_cpus=${SIMULATION_NUM_CPUS}, threads=${SIMULATION_WORKER_THREADS:-auto})"
echo "Validating metric collection"
echo "=============================================="
echo ""

# Find checkpoints
if is_enabled "$RUN_ZERO_SHOT"; then
    ZERO_SHOT_CKPT="${PLUTO_EVAL_CHECKPOINT:-$(pwd)/checkpoints/pluto_1M_aux_cil.ckpt}"
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
echo "📍 Using scenario filter: ${FILTER_NAME}"
echo "📍 Using scenario builder: ${SCENARIO_BUILDER}"
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

if is_enabled "${SKIP_RESULT_COLLECTION:-false}"; then
    echo "Skipping result collection by request."
else
"$PYTHON_BIN" ${REPO_ROOT}/scripts/evaluation/collect_quick_test_results.py \
    --tests "$COLLECT_TEST" \
    --methods "$COLLECT_METHODS" \
    --detail || echo "Could not collect ${FILTER_NAME} summary"
fi

echo ""
echo "=============================================="
echo "Next steps:"
echo "  1. Check if NR-CLS metrics are present"
echo "  2. If OK, use bash analyze.sh for later analysis reruns"
echo "  3. If not OK, debug metric configuration"
echo "=============================================="
