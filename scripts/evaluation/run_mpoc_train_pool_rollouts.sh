#!/bin/bash
# Run train-pool closed-loop rollouts for the MPOC curriculum baseline.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

TRAIN_FILTER="${TRAIN_FILTER:-uniform_train_all}"
BATCH_SIZE="${BATCH_SIZE:-200}"
PLUTO_CKPT="${PLUTO_CKPT:-${REPO_ROOT}/checkpoints/pluto_1M_aux_cil.ckpt}"
PLUTO_EXPERIMENT="${PLUTO_EXPERIMENT:-mpoc_zeroshot_pluto_train}"
IDM_EXPERIMENT="${IDM_EXPERIMENT:-mpoc_idm_train}"
MPOC_MODE="${MPOC_MODE:-full}"
PLUTO_SUBSET_FILTER="${PLUTO_SUBSET_FILTER:-mpoc_pluto_subset}"
RUN_PLUTO="${RUN_PLUTO:-true}"
RUN_IDM="${RUN_IDM:-true}"
DISABLE_SIMULATION_LOG="${DISABLE_SIMULATION_LOG:-true}"
QUIET_SIMULATION="${QUIET_SIMULATION:-false}"
SIMULATION_WORKER="${SIMULATION_WORKER:-ray_distributed}"
SIMULATION_WORKER_THREADS="${SIMULATION_WORKER_THREADS:-}"
SIMULATION_WORKER_MAX_WORKERS="${SIMULATION_WORKER_MAX_WORKERS:-}"
SIMULATION_NUM_GPUS="${SIMULATION_NUM_GPUS:-0}"
SIMULATION_NUM_CPUS="${SIMULATION_NUM_CPUS:-1}"
SIMULATION_RAY_LOG_TO_DRIVER="${SIMULATION_RAY_LOG_TO_DRIVER:-true}"

# Optional: set SCENARIO_LIMIT for a smoke run before launching the full train pool.
SCENARIO_LIMIT="${SCENARIO_LIMIT:-}"

is_enabled() {
    case "$1" in
        true|TRUE|True|1|yes|YES|Yes|on|ON|On) return 0 ;;
        *) return 1 ;;
    esac
}

build_common_args() {
    local filter_name="$1"
    local args=(
        --filter "$filter_name"
        --batch-size "$BATCH_SIZE"
        --worker "$SIMULATION_WORKER"
        --num-gpus "$SIMULATION_NUM_GPUS"
        --num-cpus "$SIMULATION_NUM_CPUS"
    )
    if [ "$SIMULATION_WORKER" = "ray_distributed" ]; then
        [ -n "$SIMULATION_WORKER_THREADS" ] && args+=(--worker-threads "$SIMULATION_WORKER_THREADS")
        [ -n "$SIMULATION_RAY_LOG_TO_DRIVER" ] && args+=(--ray-log-to-driver "$SIMULATION_RAY_LOG_TO_DRIVER")
    elif [ "$SIMULATION_WORKER" = "single_machine_thread_pool" ]; then
        [ -n "$SIMULATION_WORKER_MAX_WORKERS" ] && args+=(--worker-max-workers "$SIMULATION_WORKER_MAX_WORKERS")
    fi
    if [ -n "$SCENARIO_LIMIT" ]; then
        args+=(--limit "$SCENARIO_LIMIT")
    fi
    if is_enabled "$DISABLE_SIMULATION_LOG"; then
        args+=(--disable-simulation-log)
    fi
    if is_enabled "$QUIET_SIMULATION"; then
        args+=(--quiet --simulation-verbose false)
    fi
    printf '%s\n' "${args[@]}"
}

# Set up Python/runtime paths. Supports conda, .venv, or an already-active env.
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/env_bootstrap.sh"

echo "=============================================="
echo "MPOC train-pool closed-loop rollouts"
echo "=============================================="
echo "MPOC mode:         $MPOC_MODE"
echo "Train filter:      $TRAIN_FILTER"
if [ "$MPOC_MODE" = "idm_full_pluto_selective" ]; then
    echo "PLUTO filter:      $PLUTO_SUBSET_FILTER"
fi
echo "Batch size:        $BATCH_SIZE"
echo "PLUTO experiment:  $PLUTO_EXPERIMENT"
echo "IDM experiment:    $IDM_EXPERIMENT"
echo "Simulation logs:   $DISABLE_SIMULATION_LOG"
echo "Quiet simulation:  $QUIET_SIMULATION"
echo "Worker:            $SIMULATION_WORKER"
echo "Worker threads:    ${SIMULATION_WORKER_THREADS:-auto}"
echo "Per-sim CPU/GPU:   ${SIMULATION_NUM_CPUS}/${SIMULATION_NUM_GPUS}"
if [ -n "$SCENARIO_LIMIT" ]; then
    echo "Scenario limit:    $SCENARIO_LIMIT"
fi
echo ""

mapfile -t IDM_ARGS < <(build_common_args "$TRAIN_FILTER")

if is_enabled "$RUN_IDM"; then
    echo "Running IDM train-pool rollout..."
    python scripts/evaluation/run_simulation_batched.py \
        "${IDM_ARGS[@]}" \
        --planner idm_planner \
        --experiment "$IDM_EXPERIMENT"
fi

if is_enabled "$RUN_PLUTO"; then
    if [ ! -f "$PLUTO_CKPT" ]; then
        echo "Error: PLUTO checkpoint not found: $PLUTO_CKPT" >&2
        exit 1
    fi
    if [ "$MPOC_MODE" = "idm_full_pluto_selective" ]; then
        PLUTO_FILTER="$PLUTO_SUBSET_FILTER"
        if [ ! -f "${REPO_ROOT}/config/scenario_filter/${PLUTO_FILTER}.yaml" ]; then
            echo "Error: selective PLUTO subset filter is missing: ${REPO_ROOT}/config/scenario_filter/${PLUTO_FILTER}.yaml" >&2
            echo "Build it first with create_mpoc_filters.py --mpoc-mode idm_full_pluto_selective --build-pluto-subset-only --copy-to-pluto-config" >&2
            exit 1
        fi
    else
        PLUTO_FILTER="$TRAIN_FILTER"
    fi
    mapfile -t PLUTO_ARGS < <(build_common_args "$PLUTO_FILTER")
    echo "Running zero-shot PLUTO train-pool rollout..."
    python scripts/evaluation/run_simulation_batched.py \
        "${PLUTO_ARGS[@]}" \
        --planner pluto_planner \
        --ckpt "$PLUTO_CKPT" \
        --experiment "$PLUTO_EXPERIMENT"
fi

echo ""
echo "MPOC rollouts finished."
echo "Next: build filters from these experiment manifests with create_mpoc_filters.py."
