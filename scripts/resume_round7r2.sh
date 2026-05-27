#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
source .venv/bin/activate

WAE_ROUTER_PILOT_ROOT="${WAE_ROUTER_PILOT_ROOT:-${REPO_ROOT}}"
MASROUTER_PATH="${MASROUTER_PATH:-/workspace/masrouter}"
export WAE_ROUTER_PILOT_ROOT MASROUTER_PATH

export PYTHONPATH="${WAE_ROUTER_PILOT_ROOT}:${MASROUTER_PATH}:${PYTHONPATH:-}"
export ROUND_PREFIX=round7r2
export RUN_STAGE_A=0
export MAS_CURVE_ONLY_SEED1=1
export HARDCASE_TAU_MAIN=0.5
export HARDCASE_TAU_SWEEP='0.3 0.7'
export CONTINUE_ON_ERROR=1
export RUN_MAX_RETRIES=1

./scripts/run_round7_feedback.sh
