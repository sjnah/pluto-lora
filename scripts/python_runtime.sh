#!/usr/bin/env bash
# Resolve one Python executable for shell orchestration and training commands.

if [ -n "${PYTHON_BIN:-}" ]; then
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        echo "Error: configured PYTHON_BIN is not executable: $PYTHON_BIN" >&2
        return 1 2>/dev/null || exit 1
    fi
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
else
    echo "Error: neither python3 nor python is available in PATH." >&2
    return 1 2>/dev/null || exit 1
fi

export PYTHON_BIN
