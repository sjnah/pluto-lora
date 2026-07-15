#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

output="$(
    cd "$REPO_ROOT"
    METHOD=uniform \
    CURRICULUM_VERSION=uniform_continuous_test \
    FEATURE_CACHE_NAME= \
    DRY_RUN=true \
        bash scripts/training/run_lora_experiment.sh
)"

dry_run_count="$(grep -c '^DRY_RUN:' <<<"$output")"
[ "$dry_run_count" -eq 1 ]

grep -q 'Execution: continuous Uniform FT for 12 epochs' <<<"$output"
grep -q 'Phase transitions: disabled; optimizer reset: disabled' <<<"$output"
grep -q 'Uniform FT is a single continuous run; A/B curriculum phases are skipped.' <<<"$output"
grep -q 'experiment=curriculum_lora_uniform_only_uniform_continuous_test_flat_area_matched_v1_stage3_uniform' <<<"$output"
grep -q 'pretrained_ckpt=' <<<"$output"
grep -q 'epochs=12' <<<"$output"
grep -q 'lora.reset_optimizer_moments_on_resume=false' <<<"$output"
grep -q 'lora.execution_mode=continuous_uniform' <<<"$output"

if grep -q 'stage1_raw\|stage2_uniform\| checkpoint=' <<<"$output"; then
    echo "Uniform dry-run unexpectedly contains a staged resume path" >&2
    exit 1
fi

echo "Uniform continuous-run contract passed."
