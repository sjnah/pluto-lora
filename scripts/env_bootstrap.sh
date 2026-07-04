#!/usr/bin/env bash
# Shared runtime bootstrap for local PLUTO scripts.
# Supports either an already-active Python environment, a repo-local .venv, or
# the historical conda env when USE_CONDA is left enabled.

if [ -z "${REPO_ROOT:-}" ]; then
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
if [ -z "${WORKSPACE_ROOT:-}" ]; then
    WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
fi

NUPLAN_DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-${WORKSPACE_ROOT}/nuplan-devkit}"
if [ -z "${NUPLAN_RUNTIME_ROOT:-}" ] && [ -d "/root/vessl-nuplan" ]; then
    NUPLAN_RUNTIME_ROOT="/root/vessl-nuplan"
else
    NUPLAN_RUNTIME_ROOT="${NUPLAN_RUNTIME_ROOT:-${NUPLAN_DEVKIT_ROOT}/nuplan}"
fi
INTERPLAN_ROOT="${INTERPLAN_ROOT:-${WORKSPACE_ROOT}/interPlan}"

choose_exp_root() {
    local candidate="${NUPLAN_RUNTIME_ROOT}/exp"
    local fallback="${NUPLAN_FALLBACK_EXP_ROOT:-${WORKSPACE_ROOT}/nuplan-exp}"
    local probe=".write_test_$$"
    local explicit_exp_root="${NUPLAN_EXP_ROOT:-}"

    if [ "${NUPLAN_PRESERVE_EXPLICIT_PATHS:-0}" = "1" ] && [ -n "${NUPLAN_EXP_ROOT:-}" ]; then
        echo "$NUPLAN_EXP_ROOT"
        return
    fi

    if mkdir -p "$candidate" 2>/dev/null && ( : > "${candidate}/${probe}" ) 2>/dev/null; then
        rm -f "${candidate}/${probe}"
        echo "$candidate"
        return
    fi

    if [ -n "$explicit_exp_root" ]; then
        echo "Error: explicit NUPLAN_EXP_ROOT is not writable: $explicit_exp_root" >&2
        echo "Fix the mount permissions or choose a writable shared exp root." >&2
        exit 1
    fi

    echo "Warning: ${candidate} is not writable; falling back to ${fallback}" >&2
    mkdir -p "$fallback"
    echo "$fallback"
}

if [ "${USE_CONDA:-1}" != "0" ] && command -v conda >/dev/null 2>&1; then
    if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "${CONDA_ENV_NAME:-nuplan}" ]; then
        echo "Activating conda environment: ${CONDA_ENV_NAME:-nuplan}"
        eval "$(conda shell.bash hook)"
        conda activate "${CONDA_ENV_NAME:-nuplan}"
    fi
elif [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${WORKSPACE_ROOT}/.venv/bin/activate" ]; then
    echo "Activating virtual environment: ${WORKSPACE_ROOT}/.venv"
    # shellcheck disable=SC1091
    source "${WORKSPACE_ROOT}/.venv/bin/activate"
fi

if [ ! -d "$NUPLAN_DEVKIT_ROOT/nuplan" ]; then
    echo "Error: nuPlan devkit package not found: $NUPLAN_DEVKIT_ROOT/nuplan" >&2
    echo "Expected workspace layout: ${WORKSPACE_ROOT}/pluto and ${WORKSPACE_ROOT}/nuplan-devkit" >&2
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${NUPLAN_DEVKIT_ROOT}:${INTERPLAN_ROOT}:${PYTHONPATH:-}"
export NUPLAN_DEVKIT_ROOT
export NUPLAN_RUNTIME_ROOT
if [ "${NUPLAN_PRESERVE_EXPLICIT_PATHS:-0}" = "1" ]; then
    export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-${NUPLAN_RUNTIME_ROOT}/database}"
    export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${NUPLAN_DATA_ROOT}/maps}"
    export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-${NUPLAN_RUNTIME_ROOT}/exp}"
else
    export NUPLAN_DATA_ROOT="${NUPLAN_RUNTIME_ROOT}/database"
    export NUPLAN_MAPS_ROOT="${NUPLAN_DATA_ROOT}/maps"
    export NUPLAN_EXP_ROOT="$(choose_exp_root)"
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

if [ "${NUPLAN_VALIDATE_PATHS:-1}" = "1" ] && [ ! -e "$NUPLAN_DATA_ROOT" ]; then
    echo "Error: NUPLAN_DATA_ROOT does not exist: $NUPLAN_DATA_ROOT" >&2
    echo "Resolved NUPLAN_DEVKIT_ROOT=$NUPLAN_DEVKIT_ROOT" >&2
    echo "Resolved NUPLAN_RUNTIME_ROOT=$NUPLAN_RUNTIME_ROOT" >&2
    echo "Set NUPLAN_RUNTIME_ROOT=/root/vessl-nuplan or source the current .env.server." >&2
    exit 1
fi

if [ "${NUPLAN_VALIDATE_PATHS:-1}" = "1" ]; then
    probe=".write_test_$$"
    if ! mkdir -p "$NUPLAN_EXP_ROOT/exp" 2>/dev/null || ! ( : > "${NUPLAN_EXP_ROOT}/exp/${probe}" ) 2>/dev/null; then
        echo "Error: NUPLAN_EXP_ROOT is not writable: $NUPLAN_EXP_ROOT" >&2
        echo "Set NUPLAN_EXP_ROOT to a writable path, for example ${WORKSPACE_ROOT}/nuplan-exp." >&2
        exit 1
    fi
    rm -f "${NUPLAN_EXP_ROOT}/exp/${probe}"
fi
