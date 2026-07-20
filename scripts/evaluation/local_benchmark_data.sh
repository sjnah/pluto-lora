#!/usr/bin/env bash
# Select the local, pruned DB snapshot when it has been fully materialized.

use_local_benchmark_data() {
    local benchmark=$1
    local enabled="${PLUTO_USE_LOCAL_BENCHMARK_DB:-auto}"
    local root="${PLUTO_LOCAL_BENCHMARK_DB_ROOT:-${WORKSPACE_ROOT}/.local-data/benchmark-db}"
    local helper="${REPO_ROOT}/scripts/evaluation/prepare_local_benchmark_dbs.py"

    case "$enabled" in
        0|false|FALSE|False|no|NO|No|off|OFF|Off)
            return 0
            ;;
    esac

    if "$PYTHON_BIN" "$helper" --root "$root" --benchmarks "$benchmark" --verify >/dev/null; then
        export NUPLAN_DATA_ROOT="$root"
        export NUPLAN_MAPS_ROOT="${NUPLAN_DATA_ROOT}/maps"
        echo "Using local ${benchmark} DB snapshot: ${NUPLAN_DATA_ROOT}"
        return 0
    fi

    if [ "$enabled" = "require" ]; then
        echo "Error: required local ${benchmark} DB snapshot is unavailable: ${root}" >&2
        return 1
    fi
    echo "Local ${benchmark} DB snapshot unavailable; using shared NFS DB."
}
