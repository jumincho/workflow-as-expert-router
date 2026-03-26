#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
source "${ROOT_DIR}/.venv/bin/activate"

export PYTHONPATH="/workspace/wae_router_pilot:/workspace/masrouter:${PYTHONPATH:-}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"

CFG="${CFG:-${ROOT_DIR}/config/model_endpoints_3x7b.yaml}"
PREFIX="${PREFIX:-round5a_$(date +%m%d_%H%M%S)}"
SEED="${SEED:-17}"
STAGEA_REPS="${STAGEA_REPS:-2}"
STAGEA_TRAIN="${STAGEA_TRAIN:-24}"
STAGEA_CAL="${STAGEA_CAL:-8}"
STAGEA_TEST_HE="${STAGEA_TEST_HE:-24}"
STAGEA_BATCH_SIZE="${STAGEA_BATCH_SIZE:-2}"
STAGEA_MAX_AGENT="${STAGEA_MAX_AGENT:-3}"
STAGEA_TIMEOUT="${STAGEA_TIMEOUT:-15}"
STAGEA_ROUTER_DETERMINISTIC="${STAGEA_ROUTER_DETERMINISTIC:-1}"

COMMON=(
  --train_samples "${STAGEA_TRAIN}"
  --calibration_size "${STAGEA_CAL}"
  --calibration_mix mixed
  --test_samples_mbpp 0
  --test_samples_humaneval "${STAGEA_TEST_HE}"
  --batch_size "${STAGEA_BATCH_SIZE}"
  --max_agent "${STAGEA_MAX_AGENT}"
  --exec_timeout_s "${STAGEA_TIMEOUT}"
  --model_endpoints "${CFG}"
  --no_fallback
  --require_heterogeneous_endpoints
  --endpoint_warmup
  --endpoint_ready_retries 8
  --endpoint_ready_interval_s 10
  --deterministic_inference
  --enable_sample_trace
  --inject_tests_into_humaneval_query
  --premium_require_tests 1
  --seed "${SEED}"
)

EXTRA_ARGS=()
if [[ "${STAGEA_ROUTER_DETERMINISTIC}" == "1" ]]; then
  EXTRA_ARGS+=(--deterministic_router_components)
fi

PARETO_CACHE="/workspace/wae_router_pilot/runs/${PREFIX}_pareto_cache.json"
echo "${PREFIX}" > /workspace/wae_router_pilot/runs/round5a_latest_prefix.txt
echo "StageA prefix=${PREFIX}"
echo "StageA reps/train/cal/test_he/max_agent=${STAGEA_REPS}/${STAGEA_TRAIN}/${STAGEA_CAL}/${STAGEA_TEST_HE}/${STAGEA_MAX_AGENT}"

for REP in $(seq 1 "${STAGEA_REPS}"); do
  python -m src.run_pilot \
    --mode wae_static_cheap "${COMMON[@]}" "${EXTRA_ARGS[@]}" \
    --epochs 0 \
    --force_workflow_id wf_io_general \
    --run_id "${PREFIX}_rep${REP}_wae_static_cheap"

  python -m src.run_pilot \
    --mode wae_dynamic_no_premium "${COMMON[@]}" "${EXTRA_ARGS[@]}" \
    --epochs 0 \
    --run_id "${PREFIX}_rep${REP}_wae_dynamic_no_premium"

  python -m src.run_pilot \
    --mode wae_dynamic "${COMMON[@]}" "${EXTRA_ARGS[@]}" \
    --epochs 0 \
    --reuse_pareto_json "${PARETO_CACHE}" \
    --run_id "${PREFIX}_rep${REP}_wae_dynamic"
done

python - <<'PY'
import json
import os
from pathlib import Path

prefix = Path("/workspace/wae_router_pilot/runs/round5a_latest_prefix.txt").read_text().strip()
runs_root = "/workspace/wae_router_pilot/runs"
modes = ["wae_static_cheap", "wae_dynamic_no_premium", "wae_dynamic"]


def load_trace(path: str):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("split") != "humaneval_eval":
                continue
            rows.append(rec)
    return rows


def rel_diff(a: float, b: float) -> float:
    den = max(abs(a), 1e-12)
    return abs(a - b) / den


report = {"prefix": prefix, "analysis": {}}

for mode in modes:
    p1 = f"{runs_root}/{prefix}_rep1_{mode}/logs/sample_trace.jsonl"
    p2 = f"{runs_root}/{prefix}_rep2_{mode}/logs/sample_trace.jsonl"
    t1 = load_trace(p1)
    t2 = load_trace(p2)
    i1 = {r["query_hash"]: r for r in t1}
    i2 = {r["query_hash"]: r for r in t2}
    common = sorted(set(i1).intersection(i2))

    pass_mismatch = 0
    workflow_mismatch = 0
    retry_mismatch = 0
    role_mismatch = 0
    task_mismatch = 0
    collab_mismatch = 0
    prompt_template_mismatch = 0
    output_hash_mismatch = 0
    token_mismatch = 0
    token_rel = []
    cost_rel = []
    pass_mismatch_workflows = {}
    top_token = []
    examples = []

    for qh in common:
        a = i1[qh]
        b = i2[qh]
        if int(a.get("pass", 0)) != int(b.get("pass", 0)):
            pass_mismatch += 1
        if str(a.get("chosen_workflow_ids")) != str(b.get("chosen_workflow_ids")):
            workflow_mismatch += 1
        if str(a.get("selected_roles", [])) != str(b.get("selected_roles", [])):
            role_mismatch += 1
        if str(a.get("task", "")) != str(b.get("task", "")):
            task_mismatch += 1
        if str(a.get("collab", "")) != str(b.get("collab", "")):
            collab_mismatch += 1
        if str(a.get("prompt_template_hash", "")) != str(b.get("prompt_template_hash", "")):
            prompt_template_mismatch += 1
        if int(a.get("max_tokens_auto_reduce_retries", 0)) != int(
            b.get("max_tokens_auto_reduce_retries", 0)
        ):
            retry_mismatch += 1
        if str(a.get("output_hash", "")) != str(b.get("output_hash", "")):
            output_hash_mismatch += 1

        t_a = int(a.get("prompt_tokens", 0)) + int(a.get("completion_tokens", 0))
        t_b = int(b.get("prompt_tokens", 0)) + int(b.get("completion_tokens", 0))
        if t_a != t_b:
            token_mismatch += 1
        tr = rel_diff(float(t_a), float(t_b))
        token_rel.append(tr)

        c_a = float(a.get("sample_cost", 0.0))
        c_b = float(b.get("sample_cost", 0.0))
        cr = rel_diff(c_a, c_b)
        cost_rel.append(cr)

        top_token.append(
            {
                "query_hash": qh,
                "token_rel_diff": tr,
                "token_rep1": t_a,
                "token_rep2": t_b,
                "output_hash_rep1": str(a.get("output_hash", "")),
                "output_hash_rep2": str(b.get("output_hash", "")),
                "workflow_rep1": a.get("chosen_workflow_ids", []),
                "workflow_rep2": b.get("chosen_workflow_ids", []),
                "roles_rep1": a.get("selected_roles", []),
                "roles_rep2": b.get("selected_roles", []),
                "pass_rep1": int(a.get("pass", 0)),
                "pass_rep2": int(b.get("pass", 0)),
            }
        )

        if int(a.get("pass", 0)) != int(b.get("pass", 0)):
            wf_key = f"{a.get('chosen_workflow_ids', [])} -> {b.get('chosen_workflow_ids', [])}"
            pass_mismatch_workflows[wf_key] = int(pass_mismatch_workflows.get(wf_key, 0)) + 1

        if len(examples) < 10 and (
            int(a.get("pass", 0)) != int(b.get("pass", 0))
            or str(a.get("chosen_workflow_ids")) != str(b.get("chosen_workflow_ids"))
            or str(a.get("output_hash", "")) != str(b.get("output_hash", ""))
        ):
            examples.append({"query_hash": qh, "rep1": a, "rep2": b})

    top_token = sorted(top_token, key=lambda x: x["token_rel_diff"], reverse=True)[:10]
    token_rel_sorted = sorted(token_rel)
    cost_rel_sorted = sorted(cost_rel)
    def pct(xs, p):
        if not xs:
            return 0.0
        idx = min(len(xs) - 1, int((len(xs) - 1) * p))
        return float(xs[idx])

    token_p95_target = 0.005 if mode == "wae_dynamic" else 0.03
    gate_pass = (
        pass_mismatch == 0
        and workflow_mismatch == 0
        and retry_mismatch == 0
        and pct(token_rel_sorted, 0.95) <= token_p95_target
    )

    report["analysis"][mode] = {
        "n_rep1": len(t1),
        "n_rep2": len(t2),
        "n_common": len(common),
        "pass_mismatch": pass_mismatch,
        "workflow_mismatch": workflow_mismatch,
        "retry_mismatch": retry_mismatch,
        "role_mismatch": role_mismatch,
        "task_mismatch": task_mismatch,
        "collab_mismatch": collab_mismatch,
        "prompt_template_mismatch": prompt_template_mismatch,
        "output_hash_mismatch": output_hash_mismatch,
        "token_mismatch": token_mismatch,
        "token_rel_diff_avg": float(sum(token_rel) / max(len(token_rel), 1)),
        "token_rel_diff_p95": pct(token_rel_sorted, 0.95),
        "cost_rel_diff_avg": float(sum(cost_rel) / max(len(cost_rel), 1)),
        "cost_rel_diff_p95": pct(cost_rel_sorted, 0.95),
        "cost_mismatch_rate_gt_1pct": float(sum(1 for x in cost_rel if x > 0.01) / max(len(cost_rel), 1)),
        "cost_mismatch_rate_gt_3pct": float(sum(1 for x in cost_rel if x > 0.03) / max(len(cost_rel), 1)),
        "cost_mismatch_rate_gt_5pct": float(sum(1 for x in cost_rel if x > 0.05) / max(len(cost_rel), 1)),
        "pass_mismatch_workflows": pass_mismatch_workflows,
        "token_rel_diff_p95_target": float(token_p95_target),
        "stageA_gate_pass": bool(gate_pass),
        "top_token_rel_diff_samples": top_token,
        "examples": examples,
    }

json_path = f"{runs_root}/{prefix}_stageA_analysis_v2.json"
md_path = f"{runs_root}/{prefix}_stageA_analysis_v2.md"
Path(json_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

lines = ["# Stage A Analysis V2", "", f"- prefix: `{prefix}`", ""]
for mode, a in report["analysis"].items():
    lines.extend(
        [
            f"## {mode}",
            f"- n_common: `{a['n_common']}`",
            f"- pass_mismatch: `{a['pass_mismatch']}`",
            f"- workflow_mismatch: `{a['workflow_mismatch']}`",
            f"- retry_mismatch: `{a['retry_mismatch']}`",
            f"- role_mismatch: `{a['role_mismatch']}`",
            f"- task_mismatch: `{a['task_mismatch']}`",
            f"- collab_mismatch: `{a['collab_mismatch']}`",
            f"- prompt_template_mismatch: `{a['prompt_template_mismatch']}`",
            f"- output_hash_mismatch: `{a['output_hash_mismatch']}`",
            f"- token_mismatch: `{a['token_mismatch']}`",
            f"- token_rel_diff_avg/p95: `{a['token_rel_diff_avg']:.6f}` / `{a['token_rel_diff_p95']:.6f}`",
            f"- token_rel_diff_p95_target: `{a['token_rel_diff_p95_target']:.6f}`",
            f"- cost_rel_diff_avg/p95: `{a['cost_rel_diff_avg']:.6f}` / `{a['cost_rel_diff_p95']:.6f}`",
            f"- cost_mismatch_rate(>1%/>3%/>5%): `{a['cost_mismatch_rate_gt_1pct']:.4f}` / `{a['cost_mismatch_rate_gt_3pct']:.4f}` / `{a['cost_mismatch_rate_gt_5pct']:.4f}`",
            f"- pass_mismatch_workflows: `{a['pass_mismatch_workflows']}`",
            f"- stageA_gate_pass: `{a['stageA_gate_pass']}`",
            "",
        ]
    )
Path(md_path).write_text("\n".join(lines), encoding="utf-8")
print(json_path)
print(md_path)
PY

echo "STAGE_A_DONE prefix=${PREFIX}"
