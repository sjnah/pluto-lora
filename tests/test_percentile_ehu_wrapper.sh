#!/usr/bin/env bash

set -euo pipefail

TEST_DIR="$(mktemp -d)"
trap 'rm -rf "$TEST_DIR"' EXIT

export METHOD=llm
export TYPE_ROUTING_MODE=on
export CURRICULUM_BASE_EXP=test_phase_transition_contract
export TYPE_ROUTING_METADATA_PATH="${TEST_DIR}/source_metadata.csv"
export TYPE_ROUTING_SNAPSHOT_PATH="${TEST_DIR}/frozen_metadata.csv"

printf 'scenario_id,demonstration_type\nscene_a,normal\n' > "$TYPE_ROUTING_METADATA_PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../scripts/training" && pwd)"
# shellcheck source=../scripts/training/run_lora_experiment_percentile_ehu.sh
source "${SCRIPT_DIR}/run_lora_experiment_percentile_ehu.sh"

case "$CUMULATIVE_EXPOSURE_STATE_PATH" in
    "${REPO_ROOT}"/artifacts/curriculum_sampling/*) ;;
    *)
        echo "Expected an absolute repo-shared exposure path, got: $CUMULATIVE_EXPOSURE_STATE_PATH" >&2
        exit 1
        ;;
esac

freeze_type_routing_metadata
[ "$DEMONSTRATION_TYPE_METADATA_PATH" = "$TYPE_ROUTING_SNAPSHOT_PATH" ]
[ -n "$TYPE_ROUTING_METADATA_SHA256" ]
cmp -s "$TYPE_ROUTING_METADATA_PATH" "$TYPE_ROUTING_SNAPSHOT_PATH"

# The frozen copy remains sufficient after the upstream artifact disappears.
rm "$TYPE_ROUTING_METADATA_PATH"
verify_type_routing_metadata_snapshot

# A changed frozen file is detected before the next phase starts.
printf 'corrupted\n' >> "$TYPE_ROUTING_SNAPSHOT_PATH"
if (verify_type_routing_metadata_snapshot) 2>/dev/null; then
    echo "Expected checksum verification to reject changed metadata" >&2
    exit 1
fi

# Phase transitions must select the experiment-local post-fit checkpoint, not
# the Hydra-run-level callback checkpoint.
CHECKPOINT_ROOT="${TEST_DIR}/checkpoint_lookup"
mkdir -p \
    "${CHECKPOINT_ROOT}/outputs/2026-01-01/00-00-00/checkpoints" \
    "${CHECKPOINT_ROOT}/outputs/2026-01-01/00-00-00/outputs/phase_exp/checkpoints" \
    "${CHECKPOINT_ROOT}/outputs/2026-01-02/00-00-00/outputs/phase_exp/checkpoints"
touch "${CHECKPOINT_ROOT}/outputs/2026-01-01/00-00-00/checkpoints/last.ckpt"
touch "${CHECKPOINT_ROOT}/outputs/2026-01-01/00-00-00/outputs/phase_exp/checkpoints/last.ckpt"
touch "${CHECKPOINT_ROOT}/outputs/2026-01-02/00-00-00/outputs/phase_exp/checkpoints/last.ckpt"

pushd "$CHECKPOINT_ROOT" >/dev/null
selected_checkpoint="$(find_latest_checkpoint phase_exp)"
popd >/dev/null

expected_checkpoint="${CHECKPOINT_ROOT}/outputs/2026-01-02/00-00-00/outputs/phase_exp/checkpoints/last.ckpt"
[ "$selected_checkpoint" = "$expected_checkpoint" ] || {
    echo "Unexpected phase checkpoint: $selected_checkpoint" >&2
    exit 1
}
