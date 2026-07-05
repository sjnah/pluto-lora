#!/usr/bin/env bash
# Run the selective/adaptive MPOC pipeline end-to-end.
#
# This wrapper keeps the existing PLUTO/llm-taxonomy entrypoints unchanged and
# simply executes the four required stages in order.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PLUTO_ROOT="${WORKSPACE_ROOT}/pluto"
LLM_TAXONOMY_ROOT="${WORKSPACE_ROOT}/llm-taxonomy"

IDM_EXPERIMENT="${IDM_EXPERIMENT:-mpoc_idm_train}"
PLUTO_EXPERIMENT="${PLUTO_EXPERIMENT:-mpoc_zeroshot_pluto_selective}"
MPOC_MODE="${MPOC_MODE:-idm_full_pluto_selective}"
TRAIN_FILTER="${TRAIN_FILTER:-uniform_train_all}"
BATCH_SIZE="${BATCH_SIZE:-200}"
PLUTO_BUDGET="${PLUTO_BUDGET:-900}"

echo "=============================================="
echo "Selective MPOC pipeline"
echo "=============================================="
echo "Workspace:        ${WORKSPACE_ROOT}"
echo "Train filter:     ${TRAIN_FILTER}"
echo "IDM experiment:   ${IDM_EXPERIMENT}"
echo "PLUTO experiment: ${PLUTO_EXPERIMENT}"
echo "Batch size:       ${BATCH_SIZE}"
echo "PLUTO budget:     ${PLUTO_BUDGET}"
echo ""

echo "=============================================="
echo "Stage 1/4: IDM rollout over full FT train pool"
echo "=============================================="
cd "${PLUTO_ROOT}"
RUN_PLUTO=false \
RUN_IDM=true \
MPOC_MODE="${MPOC_MODE}" \
TRAIN_FILTER="${TRAIN_FILTER}" \
BATCH_SIZE="${BATCH_SIZE}" \
IDM_EXPERIMENT="${IDM_EXPERIMENT}" \
bash scripts/evaluation/run_mpoc_train_pool_rollouts.sh

echo ""
echo "=============================================="
echo "Stage 2/4: Build PLUTO subset from IDM outcomes"
echo "=============================================="
cd "${LLM_TAXONOMY_ROOT}"
python scripts/experiments/pluto/create_mpoc_filters.py \
  --mpoc-mode "${MPOC_MODE}" \
  --train-filter "${PLUTO_ROOT}/config/scenario_filter/${TRAIN_FILTER}.yaml" \
  --idm-experiment "${IDM_EXPERIMENT}" \
  --pluto-budget "${PLUTO_BUDGET}" \
  --build-pluto-subset-only \
  --copy-to-pluto-config

echo ""
echo "=============================================="
echo "Stage 3/4: PLUTO rollout on selected subset"
echo "=============================================="
cd "${PLUTO_ROOT}"
RUN_IDM=false \
RUN_PLUTO=true \
MPOC_MODE="${MPOC_MODE}" \
BATCH_SIZE="${BATCH_SIZE}" \
PLUTO_EXPERIMENT="${PLUTO_EXPERIMENT}" \
bash scripts/evaluation/run_mpoc_train_pool_rollouts.sh

echo ""
echo "=============================================="
echo "Stage 4/4: Build final selective MPOC curriculum"
echo "=============================================="
cd "${LLM_TAXONOMY_ROOT}"
python scripts/experiments/pluto/create_mpoc_filters.py \
  --mpoc-mode "${MPOC_MODE}" \
  --train-filter "${PLUTO_ROOT}/config/scenario_filter/${TRAIN_FILTER}.yaml" \
  --idm-experiment "${IDM_EXPERIMENT}" \
  --pluto-experiment "${PLUTO_EXPERIMENT}" \
  --pluto-budget "${PLUTO_BUDGET}" \
  --copy-to-pluto-config

echo ""
echo "=============================================="
echo "Selective MPOC pipeline complete"
echo "=============================================="
echo "Curriculum artifacts: ${LLM_TAXONOMY_ROOT}/artifacts/scenario_filters/pluto_mpoc"
echo "Active PLUTO filters: ${PLUTO_ROOT}/config/scenario_filter/mpoc_train_*.yaml"
