#!/usr/bin/env bash
set -euo pipefail

cd /workspace/wae_router_pilot
source .venv/bin/activate

export PYTHONPATH=/workspace/wae_router_pilot:/workspace/masrouter:${PYTHONPATH:-}
export ROUND_PREFIX=round7r2
export RUN_STAGE_A=0
export MAS_CURVE_ONLY_SEED1=1
export HARDCASE_TAU_MAIN=0.5
export HARDCASE_TAU_SWEEP='0.3 0.7'
export CONTINUE_ON_ERROR=1
export RUN_MAX_RETRIES=1

./scripts/run_round7_feedback.sh
