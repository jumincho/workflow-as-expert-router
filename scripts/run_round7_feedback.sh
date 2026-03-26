#!/usr/bin/env bash
set -u -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
source "${ROOT_DIR}/.venv/bin/activate"

export PYTHONPATH="/workspace/wae_router_pilot:/workspace/masrouter:${PYTHONPATH:-}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

CFG="${CFG:-${ROOT_DIR}/config/model_endpoints_3x7b.yaml}"
ROUND_PREFIX="${ROUND_PREFIX:-round7r1}"
export ROUND_PREFIX
SEEDS="${SEEDS:-1 2 3}"

TRAIN="${TRAIN:-64}"
CAL="${CAL:-64}"
TEST_MBPP="${TEST_MBPP:-80}"
TEST_HE="${TEST_HE:-80}"
EPOCHS="${EPOCHS:-1}"
BS="${BS:-2}"
MAX_AGENT="${MAX_AGENT:-3}"
TIMEOUT="${TIMEOUT:-15}"
ROUTER_CUDA_VISIBLE_DEVICES="${ROUTER_CUDA_VISIBLE_DEVICES:-3}"
export CUDA_VISIBLE_DEVICES="${ROUTER_CUDA_VISIBLE_DEVICES}"

CHEAP_WF="${CHEAP_WF:-wf_io_general}"
PREMIUM_WF="${PREMIUM_WF:-wf_gen3_test_select_general}"
HARDCASE_PREMIUM_WF="${HARDCASE_PREMIUM_WF:-wf_gen2_test_select_general}"
HARDCASE_TAU_MAIN="${HARDCASE_TAU_MAIN:-0.5}"
HARDCASE_TAU_SWEEP="${HARDCASE_TAU_SWEEP:-0.3 0.7}"

BUDGET_POINTS="${BUDGET_POINTS:-0.00026,0.00030,0.00040}"
DYNAMIC_ENTROPY_REG="${DYNAMIC_ENTROPY_REG:-0.02}"
DYNAMIC_NO_PREMIUM_ENTROPY_REG="${DYNAMIC_NO_PREMIUM_ENTROPY_REG:-0.0}"

MAS_CHEAP_COST_RATE="${MAS_CHEAP_COST_RATE:-700}"
MAS_BALANCED_COST_RATE="${MAS_BALANCED_COST_RATE:-300}"
MAS_PREMIUM_COST_RATE="${MAS_PREMIUM_COST_RATE:-120}"
MAS_CHEAP_MAX_AGENT="${MAS_CHEAP_MAX_AGENT:-2}"
MAS_BALANCED_MAX_AGENT="${MAS_BALANCED_MAX_AGENT:-3}"
MAS_PREMIUM_MAX_AGENT="${MAS_PREMIUM_MAX_AGENT:-4}"
MAS_CURVE_ONLY_SEED1="${MAS_CURVE_ONLY_SEED1:-1}"

RUN_STAGE_A="${RUN_STAGE_A:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
RUN_MAX_RETRIES="${RUN_MAX_RETRIES:-1}"
MONITOR_INTERVAL_S="${MONITOR_INTERVAL_S:-120}"
STALL_TIMEOUT_S="${STALL_TIMEOUT_S:-1800}"
AUTO_RELEASE_GPU_ON_EXIT="${AUTO_RELEASE_GPU_ON_EXIT:-1}"
VLLM_RELEASE_PORTS="${VLLM_RELEASE_PORTS:-8000 8001 8002}"
GPU_RELEASE_GRACE_S="${GPU_RELEASE_GRACE_S:-8}"

RUNS_ROOT="/workspace/wae_router_pilot/runs"
mkdir -p "${RUNS_ROOT}"
ORCH_LOG="${RUNS_ROOT}/${ROUND_PREFIX}_orchestrator.log"
FAIL_LOG="${RUNS_ROOT}/${ROUND_PREFIX}_failures.log"

ts() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

log() {
  local msg="$*"
  echo "[$(ts)] ${msg}" | tee -a "${ORCH_LOG}"
}

release_gpu_servers() {
  local pids=""
  for port in ${VLLM_RELEASE_PORTS}; do
    local found
    found="$(pgrep -f "vllm.entrypoints.openai.api_server.*--port ${port}( |$)" || true)"
    if [[ -n "${found}" ]]; then
      pids+=$'\n'"${found}"
    fi
  done
  pids="$(echo "${pids}" | tr ' ' '\n' | sed '/^$/d' | sort -u)"
  if [[ -z "${pids}" ]]; then
    log "GPU release: no vLLM api_server processes matched ports [${VLLM_RELEASE_PORTS}]"
    return 0
  fi

  local pid_line
  pid_line="$(echo "${pids}" | tr '\n' ' ')"
  log "GPU release: stopping vLLM pids=${pid_line}"
  while read -r pid; do
    [[ -n "${pid}" ]] || continue
    kill -TERM "${pid}" >/dev/null 2>&1 || true
  done <<< "${pids}"

  sleep "${GPU_RELEASE_GRACE_S}"
  while read -r pid; do
    [[ -n "${pid}" ]] || continue
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill -KILL "${pid}" >/dev/null 2>&1 || true
    fi
  done <<< "${pids}"
  log "GPU release: completed"
}

on_exit_cleanup() {
  local rc=$?
  trap - EXIT
  if [[ "${AUTO_RELEASE_GPU_ON_EXIT}" == "1" ]]; then
    release_gpu_servers || true
  fi
  return "${rc}"
}
trap on_exit_cleanup EXIT

is_run_complete() {
  local run_dir="$1"
  [[ -f "${run_dir}/metrics/summary.json" && -f "${run_dir}/report.md" ]]
}

archive_incomplete_run() {
  local run_id="$1"
  local run_dir="${RUNS_ROOT}/${run_id}"
  if [[ -d "${run_dir}" ]] && ! is_run_complete "${run_dir}"; then
    local archived="${run_dir}_failed_$(date -u '+%H%M%S')"
    mv "${run_dir}" "${archived}"
    log "Archived incomplete run: ${run_dir} -> ${archived}"
  fi
}

monitor_run_pid() {
  local pid="$1"
  local run_id="$2"
  local run_dir="${RUNS_ROOT}/${run_id}"
  local status_path="${run_dir}/logs/status.json"
  while kill -0 "${pid}" >/dev/null 2>&1; do
    if [[ -f "${status_path}" ]]; then
      local now_ts
      local st_mtime
      local age
      now_ts="$(date +%s)"
      st_mtime="$(stat -c %Y "${status_path}" 2>/dev/null || echo 0)"
      age=$((now_ts - st_mtime))
      local stage
      stage="$(python - "${status_path}" <<'PY'
import json,sys
p=sys.argv[1]
try:
    j=json.load(open(p,encoding='utf-8'))
    print(j.get('stage',''), j.get('event',''), j.get('updated_at',''))
except Exception:
    print('unknown unknown unknown')
PY
)"
      local gpu
      gpu="$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ';' || true)"
      log "watch run_id=${run_id} pid=${pid} age_s=${age} stage=${stage} gpu=[${gpu}]"
      if (( age > STALL_TIMEOUT_S )); then
        log "STALL detected run_id=${run_id} age_s=${age} > ${STALL_TIMEOUT_S}; terminating pid=${pid}"
        kill -TERM "${pid}" >/dev/null 2>&1 || true
        sleep 10
        kill -KILL "${pid}" >/dev/null 2>&1 || true
        return 124
      fi
    else
      log "watch run_id=${run_id} pid=${pid} status.json not found yet"
    fi
    sleep "${MONITOR_INTERVAL_S}"
  done
  return 0
}

run_pilot_step() {
  local run_id="$1"
  shift
  local run_dir="${RUNS_ROOT}/${run_id}"

  if is_run_complete "${run_dir}"; then
    log "SKIP completed run_id=${run_id}"
    return 0
  fi

  archive_incomplete_run "${run_id}"

  local attempt=0
  while true; do
    attempt=$((attempt + 1))
    log "START run_id=${run_id} attempt=${attempt}"
    "$@" --run_id "${run_id}" >>"${ORCH_LOG}" 2>&1 &
    local pid=$!
    monitor_run_pid "${pid}" "${run_id}" >>"${ORCH_LOG}" 2>&1 &
    local mon_pid=$!

    wait "${pid}"
    local rc=$?
    kill "${mon_pid}" >/dev/null 2>&1 || true
    wait "${mon_pid}" >/dev/null 2>&1 || true

    if [[ ${rc} -eq 0 ]] && is_run_complete "${run_dir}"; then
      log "DONE run_id=${run_id}"
      return 0
    fi

    echo "[$(ts)] run_id=${run_id} attempt=${attempt} rc=${rc}" >>"${FAIL_LOG}"
    log "FAIL run_id=${run_id} attempt=${attempt} rc=${rc}"

    if (( attempt <= RUN_MAX_RETRIES )); then
      archive_incomplete_run "${run_id}"
      log "RETRY run_id=${run_id}"
      continue
    fi

    if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
      log "CONTINUE_ON_ERROR=1 -> continue after failure run_id=${run_id}"
      return 0
    fi
    return "${rc}"
  done
}

run_compare_step() {
  local label="$1"
  shift
  log "START compare label=${label}"
  "$@" >>"${ORCH_LOG}" 2>&1
  local rc=$?
  if [[ ${rc} -ne 0 ]]; then
    echo "[$(ts)] compare label=${label} rc=${rc}" >>"${FAIL_LOG}"
    log "FAIL compare label=${label} rc=${rc}"
    if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
      return 0
    fi
    return "${rc}"
  fi
  log "DONE compare label=${label}"
  return 0
}

COMMON_ARGS=(
  --train_samples "${TRAIN}"
  --calibration_size "${CAL}"
  --calibration_mix mixed
  --test_samples_mbpp "${TEST_MBPP}"
  --test_samples_humaneval "${TEST_HE}"
  --batch_size "${BS}"
  --max_agent "${MAX_AGENT}"
  --exec_timeout_s "${TIMEOUT}"
  --model_endpoints "${CFG}"
  --no_fallback
  --require_heterogeneous_endpoints
  --endpoint_warmup
  --endpoint_ready_retries 8
  --endpoint_ready_interval_s 15
  --deterministic_inference
  --enable_sample_trace
  --inject_tests_into_humaneval_query
  --premium_require_tests 1
)

log "Round7 start prefix=${ROUND_PREFIX}"
log "CFG=${CFG}"
log "SEEDS=${SEEDS}"
log "TRAIN/CAL/TEST_MBPP/TEST_HE=${TRAIN}/${CAL}/${TEST_MBPP}/${TEST_HE}"
log "ROUTER_CUDA_VISIBLE_DEVICES=${ROUTER_CUDA_VISIBLE_DEVICES}"

if [[ "${RUN_STAGE_A}" == "1" ]]; then
  log "Running Stage A"
  "${ROOT_DIR}/scripts/run_round5_stageA.sh" >>"${ORCH_LOG}" 2>&1
  rc=$?
  if [[ ${rc} -ne 0 ]]; then
    echo "[$(ts)] stageA rc=${rc}" >>"${FAIL_LOG}"
    log "FAIL Stage A rc=${rc}"
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit "${rc}"
    fi
  fi
fi

for SEED in ${SEEDS}; do
  PREFIX="${ROUND_PREFIX}_s${SEED}"
  PARETO_CACHE="${RUNS_ROOT}/${PREFIX}_pareto_cache.json"
  log "== seed ${SEED} (${PREFIX}) =="

  run_pilot_step "${PREFIX}_masrouter_balanced" \
    env CUDA_VISIBLE_DEVICES="${ROUTER_CUDA_VISIBLE_DEVICES}" python -m src.run_pilot \
      --mode masrouter "${COMMON_ARGS[@]}" \
      --epochs "${EPOCHS}" \
      --seed "${SEED}" \
      --cost_rate "${MAS_BALANCED_COST_RATE}" \
      --max_agent "${MAS_BALANCED_MAX_AGENT}"

  if [[ "${MAS_CURVE_ONLY_SEED1}" != "1" || "${SEED}" == "1" ]]; then
    run_pilot_step "${PREFIX}_masrouter_cheap" \
      env CUDA_VISIBLE_DEVICES="${ROUTER_CUDA_VISIBLE_DEVICES}" python -m src.run_pilot \
        --mode masrouter "${COMMON_ARGS[@]}" \
        --epochs "${EPOCHS}" \
        --seed "${SEED}" \
        --cost_rate "${MAS_CHEAP_COST_RATE}" \
        --max_agent "${MAS_CHEAP_MAX_AGENT}"

    run_pilot_step "${PREFIX}_masrouter_premium" \
      env CUDA_VISIBLE_DEVICES="${ROUTER_CUDA_VISIBLE_DEVICES}" python -m src.run_pilot \
        --mode masrouter "${COMMON_ARGS[@]}" \
        --epochs "${EPOCHS}" \
        --seed "${SEED}" \
        --cost_rate "${MAS_PREMIUM_COST_RATE}" \
        --max_agent "${MAS_PREMIUM_MAX_AGENT}"
  fi

  run_pilot_step "${PREFIX}_wae_static_cheap" \
    env CUDA_VISIBLE_DEVICES="${ROUTER_CUDA_VISIBLE_DEVICES}" python -m src.run_pilot \
      --mode wae_static_cheap "${COMMON_ARGS[@]}" \
      --epochs 0 \
      --seed "${SEED}" \
      --force_workflow_id "${CHEAP_WF}"

  run_pilot_step "${PREFIX}_wae_static_premium" \
    env CUDA_VISIBLE_DEVICES="${ROUTER_CUDA_VISIBLE_DEVICES}" python -m src.run_pilot \
      --mode wae_static_premium "${COMMON_ARGS[@]}" \
      --epochs 0 \
      --seed "${SEED}" \
      --force_workflow_id "${PREMIUM_WF}"

  run_pilot_step "${PREFIX}_wae_dynamic_no_premium" \
    env CUDA_VISIBLE_DEVICES="${ROUTER_CUDA_VISIBLE_DEVICES}" python -m src.run_pilot \
      --mode wae_dynamic_no_premium "${COMMON_ARGS[@]}" \
      --epochs "${EPOCHS}" \
      --seed "${SEED}" \
      --workflow_entropy_reg "${DYNAMIC_NO_PREMIUM_ENTROPY_REG}"

  run_pilot_step "${PREFIX}_wae_dynamic" \
    env CUDA_VISIBLE_DEVICES="${ROUTER_CUDA_VISIBLE_DEVICES}" python -m src.run_pilot \
      --mode wae_dynamic "${COMMON_ARGS[@]}" \
      --epochs "${EPOCHS}" \
      --seed "${SEED}" \
      --workflow_entropy_reg "${DYNAMIC_ENTROPY_REG}" \
      --reuse_pareto_json "${PARETO_CACHE}"

  run_pilot_step "${PREFIX}_wae_dynamic_hardcase_gate" \
    env CUDA_VISIBLE_DEVICES="${ROUTER_CUDA_VISIBLE_DEVICES}" python -m src.run_pilot \
      --mode wae_dynamic_hardcase_gate "${COMMON_ARGS[@]}" \
      --epochs 0 \
      --seed "${SEED}" \
      --hardcase_tau "${HARDCASE_TAU_MAIN}" \
      --hardcase_premium_workflow_id "${HARDCASE_PREMIUM_WF}" \
      --hardcase_calibration_size "${CAL}"

  run_pilot_step "${PREFIX}_wae_cheap_first_escalate" \
    env CUDA_VISIBLE_DEVICES="${ROUTER_CUDA_VISIBLE_DEVICES}" python -m src.run_pilot \
      --mode wae_cheap_first_escalate "${COMMON_ARGS[@]}" \
      --epochs 0 \
      --seed "${SEED}" \
      --cheap_workflow_id "${CHEAP_WF}" \
      --premium_workflow_id "${PREMIUM_WF}"

  run_pilot_step "${PREFIX}_wae_dynamic_control_forced_io_general" \
    env CUDA_VISIBLE_DEVICES="${ROUTER_CUDA_VISIBLE_DEVICES}" python -m src.run_pilot \
      --mode wae_dynamic "${COMMON_ARGS[@]}" \
      --epochs "${EPOCHS}" \
      --seed "${SEED}" \
      --workflow_entropy_reg "${DYNAMIC_ENTROPY_REG}" \
      --force_workflow_id "${CHEAP_WF}"

  if [[ "${SEED}" == "1" ]]; then
    for TAU in ${HARDCASE_TAU_SWEEP}; do
      TAU_TAG="$(echo "${TAU}" | tr '.' 'p')"
      run_pilot_step "${PREFIX}_wae_dynamic_hardcase_gate_tau${TAU_TAG}" \
        env CUDA_VISIBLE_DEVICES="${ROUTER_CUDA_VISIBLE_DEVICES}" python -m src.run_pilot \
          --mode wae_dynamic_hardcase_gate "${COMMON_ARGS[@]}" \
          --epochs 0 \
          --seed "${SEED}" \
          --hardcase_tau "${TAU}" \
          --hardcase_premium_workflow_id "${HARDCASE_PREMIUM_WF}" \
          --hardcase_calibration_size "${CAL}"
    done
  fi

  CURVE_RUNS=""
  if is_run_complete "${RUNS_ROOT}/${PREFIX}_masrouter_cheap"; then
    CURVE_RUNS="masrouter_cheap=${RUNS_ROOT}/${PREFIX}_masrouter_cheap,"
  fi
  CURVE_RUNS="${CURVE_RUNS}masrouter_balanced=${RUNS_ROOT}/${PREFIX}_masrouter_balanced"
  if is_run_complete "${RUNS_ROOT}/${PREFIX}_masrouter_premium"; then
    CURVE_RUNS="${CURVE_RUNS},masrouter_premium=${RUNS_ROOT}/${PREFIX}_masrouter_premium"
  fi

  run_compare_step "${PREFIX}_compare_dynamic" \
    python -m src.compare_runs \
      --masrouter_run "${RUNS_ROOT}/${PREFIX}_masrouter_balanced" \
      --wae_dynamic_run "${RUNS_ROOT}/${PREFIX}_wae_dynamic" \
      --wae_static_cheap_run "${RUNS_ROOT}/${PREFIX}_wae_static_cheap" \
      --wae_static_premium_run "${RUNS_ROOT}/${PREFIX}_wae_static_premium" \
      --wae_dynamic_no_premium_run "${RUNS_ROOT}/${PREFIX}_wae_dynamic_no_premium" \
      --wae_cheap_first_escalate_run "${RUNS_ROOT}/${PREFIX}_wae_cheap_first_escalate" \
      --wae_dynamic_control_forced_io_run "${RUNS_ROOT}/${PREFIX}_wae_dynamic_control_forced_io_general" \
      --wae_dynamic_hardcase_gate_run "${RUNS_ROOT}/${PREFIX}_wae_dynamic_hardcase_gate" \
      --masrouter_curve_runs "${CURVE_RUNS}" \
      --target_mode wae_dynamic \
      --iso_tolerance 0.05 \
      --budget_points "${BUDGET_POINTS}" \
      --out_prefix "${RUNS_ROOT}/${PREFIX}_compare_dynamic"

  run_compare_step "${PREFIX}_compare_hardcase" \
    python -m src.compare_runs \
      --masrouter_run "${RUNS_ROOT}/${PREFIX}_masrouter_balanced" \
      --wae_dynamic_run "${RUNS_ROOT}/${PREFIX}_wae_dynamic" \
      --wae_static_cheap_run "${RUNS_ROOT}/${PREFIX}_wae_static_cheap" \
      --wae_static_premium_run "${RUNS_ROOT}/${PREFIX}_wae_static_premium" \
      --wae_dynamic_no_premium_run "${RUNS_ROOT}/${PREFIX}_wae_dynamic_no_premium" \
      --wae_cheap_first_escalate_run "${RUNS_ROOT}/${PREFIX}_wae_cheap_first_escalate" \
      --wae_dynamic_control_forced_io_run "${RUNS_ROOT}/${PREFIX}_wae_dynamic_control_forced_io_general" \
      --wae_dynamic_hardcase_gate_run "${RUNS_ROOT}/${PREFIX}_wae_dynamic_hardcase_gate" \
      --masrouter_curve_runs "${CURVE_RUNS}" \
      --target_mode wae_dynamic_hardcase_gate \
      --iso_tolerance 0.05 \
      --budget_points "${BUDGET_POINTS}" \
      --out_prefix "${RUNS_ROOT}/${PREFIX}_compare_hardcase"
done

python - <<'PY'
import glob
import json
import os
from pathlib import Path

runs_root = "/workspace/wae_router_pilot/runs"
prefix = os.environ.get("ROUND_PREFIX", "round7r1")
seed_runs = sorted(glob.glob(f"{runs_root}/{prefix}_s*_compare_dynamic.json"))
rows = []
for p in seed_runs:
    j = json.load(open(p, encoding="utf-8"))
    seed = Path(p).name.split("_")[1]
    for ds in ["mbpp_eval", "humaneval_eval"]:
        rows.append(
            {
                "seed": seed,
                "dataset": ds,
                "target": "wae_dynamic",
                "pass": bool(j["verdict"][ds]["pass"]),
                "reason": str(j["verdict"][ds]["reason"]),
                "delta_acc": float(j["iso_cost"][ds]["delta_acc"]),
                "method": str(j["iso_cost"][ds]["method"]),
            }
        )

seed_runs_h = sorted(glob.glob(f"{runs_root}/{prefix}_s*_compare_hardcase.json"))
for p in seed_runs_h:
    j = json.load(open(p, encoding="utf-8"))
    seed = Path(p).name.split("_")[1]
    for ds in ["mbpp_eval", "humaneval_eval"]:
        rows.append(
            {
                "seed": seed,
                "dataset": ds,
                "target": "wae_dynamic_hardcase_gate",
                "pass": bool(j["verdict"][ds]["pass"]),
                "reason": str(j["verdict"][ds]["reason"]),
                "delta_acc": float(j["iso_cost"][ds]["delta_acc"]),
                "method": str(j["iso_cost"][ds]["method"]),
            }
        )

report_path = f"{runs_root}/{prefix}_summary.md"
lines = ["# Round7 Summary", ""]
if not rows:
    lines.append("- No compare artifacts found.")
else:
    lines.append("| seed | dataset | target | pass | reason | delta_acc | method |")
    lines.append("|---|---|---|---:|---|---:|---|")
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['dataset']} | {r['target']} | {str(r['pass'])} | {r['reason']} | {r['delta_acc']:+.4f} | {r['method']} |"
        )
Path(report_path).write_text("\n".join(lines), encoding="utf-8")
print(report_path)
PY

log "Round7 completed"
if [[ -f "${FAIL_LOG}" ]]; then
  log "Failure log: ${FAIL_LOG}"
fi
