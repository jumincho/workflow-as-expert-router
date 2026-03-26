#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
source "${ROOT_DIR}/.venv/bin/activate"

export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export PYTHONPATH="/workspace/wae_router_pilot:/workspace/masrouter:${PYTHONPATH:-}"

STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
CFG="${ROOT_DIR}/config/model_endpoints_3x7b.yaml"

COMMON_ARGS=(
  --train_samples 24
  --test_samples_mbpp 24
  --test_samples_humaneval 24
  --epochs 1
  --batch_size 2
  --max_agent 3
  --calibration_size 24
  --calibration_mix mixed
  --exec_timeout_s 10
  --model_endpoints "${CFG}"
  --no_fallback
  --require_heterogeneous_endpoints
  --endpoint_warmup
  --endpoint_ready_retries 8
  --endpoint_ready_interval_s 15
)

python -m src.run_pilot --mode masrouter "${COMMON_ARGS[@]}" --run_id "pilot7b_masrouter_${STAMP}"
python -m src.run_pilot --mode wae_dynamic "${COMMON_ARGS[@]}" --run_id "pilot7b_wae_dynamic_${STAMP}"
python -m src.run_pilot --mode wae_static_cheap "${COMMON_ARGS[@]}" --run_id "pilot7b_wae_static_cheap_${STAMP}"
python -m src.run_pilot --mode wae_static_premium "${COMMON_ARGS[@]}" --force_workflow_id wf_gen3_test_select_coder --run_id "pilot7b_wae_static_premium_${STAMP}"

python -m src.compare_runs \
  --masrouter_run "/workspace/wae_router_pilot/runs/pilot7b_masrouter_${STAMP}" \
  --wae_dynamic_run "/workspace/wae_router_pilot/runs/pilot7b_wae_dynamic_${STAMP}" \
  --wae_static_cheap_run "/workspace/wae_router_pilot/runs/pilot7b_wae_static_cheap_${STAMP}" \
  --wae_static_premium_run "/workspace/wae_router_pilot/runs/pilot7b_wae_static_premium_${STAMP}" \
  --iso_tolerance 0.05 \
  --out_prefix "/workspace/wae_router_pilot/runs/pilot7b_compare_${STAMP}"

echo "Completed expanded run set stamp=${STAMP}"
