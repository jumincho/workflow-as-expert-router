"""Per-round diagnostic: where exactly does dynamic disagree with cheap?

When the headline iso-cost test (`compare_runs`) said "dynamic and the
cheap static baseline are roughly tied", we needed to know *why* — the
"feedback-driven diagnostics" stage of every round. This file is that.

For a run prefix like `round6r1` or `round7r2` and seeds 1/2/3 it pulls
each seed's `logs/sample_trace.jsonl` from the configured runs and computes:

- **Confusion table** on MBPP between `wae_dynamic` and `wae_static_cheap`:
  `both_pass`, `both_fail`, `dynamic_only_pass`, `static_only_pass`. The
  last two cells are the diagnostic ones — they say where dynamic actually
  bought something and where it actively hurt.
- **Top-k hard cases** where dynamic failed and cheap passed, ordered by
  retry pressure and per-sample cost — the cases that should have been
  routed to the premium workflow but were not.
- **Forced-IO path differences**: for the `wae_dynamic_control_forced_io_general`
  control, list queries where dynamic's chosen workflow / endpoint-call count
  / retry count differed from the forced-IO baseline.
- **Selection histograms**: which workflow id dynamic actually picked, by
  dataset.
- **Oracle headroom**: assuming a perfect picker between cheap and premium
  static workflows, how much better than cheap would routing be? This is
  the ceiling that dynamic was implicitly chasing.

Writes a JSON + a markdown summary side-by-side under `<output_prefix>.json/.md`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def idx_by_query_hash(rows: List[Dict[str, object]], split: str) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for r in rows:
        if str(r.get("split", "")) != split:
            continue
        qh = str(r.get("query_hash", ""))
        if qh:
            out[qh] = r
    return out


def wf_of(row: Dict[str, object]) -> str:
    c = row.get("chosen_workflow_id")
    if isinstance(c, list):
        if not c:
            return ""
        return str(c[-1])
    return str(c or "")


def parse_modes(
    runs_root: Path, prefix: str, seed: int, mode: str
) -> List[Dict[str, object]]:
    p = runs_root / f"{prefix}_s{seed}_{mode}" / "logs" / "sample_trace.jsonl"
    return load_jsonl(p)


def summarize_selection(rows: List[Dict[str, object]], split: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        if str(r.get("split", "")) != split:
            continue
        w = wf_of(r)
        out[w] = out.get(w, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


def oracle_from_static(
    cheap_rows: List[Dict[str, object]],
    premium_rows: List[Dict[str, object]],
    split: str,
) -> Dict[str, float]:
    c_idx = idx_by_query_hash(cheap_rows, split)
    p_idx = idx_by_query_hash(premium_rows, split)
    keys = sorted(set(c_idx.keys()) & set(p_idx.keys()))
    if not keys:
        return {"n": 0, "cheap_acc": 0.0, "premium_acc": 0.0, "oracle_acc": 0.0, "oracle_gain_vs_cheap": 0.0}
    cheap_ok = 0
    premium_ok = 0
    oracle_ok = 0
    for k in keys:
        c = int(c_idx[k].get("pass", 0))
        p = int(p_idx[k].get("pass", 0))
        cheap_ok += c
        premium_ok += p
        oracle_ok += 1 if (c or p) else 0
    n = len(keys)
    cheap_acc = cheap_ok / n
    premium_acc = premium_ok / n
    oracle_acc = oracle_ok / n
    return {
        "n": n,
        "cheap_acc": cheap_acc,
        "premium_acc": premium_acc,
        "oracle_acc": oracle_acc,
        "oracle_gain_vs_cheap": oracle_acc - cheap_acc,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs_root",
        default=os.environ.get(
            "WAE_RUNS_ROOT",
            f"{os.environ.get('WAE_ROUTER_PILOT_ROOT', '/workspace/wae_router_pilot')}/runs",
        ),
    )
    ap.add_argument("--prefix", required=True, help="e.g., round6r1")
    ap.add_argument("--output_prefix", default="")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    prefix = str(args.prefix)
    out_prefix = str(args.output_prefix or (runs_root / f"{prefix}_p0_analysis"))

    report: Dict[str, object] = {"prefix": prefix, "seeds": {}, "aggregate": {}}
    agg_conf = {"both_pass": 0, "both_fail": 0, "dynamic_only_pass": 0, "static_only_pass": 0}
    agg_oracle_mbpp = []
    agg_oracle_he = []

    for seed in [1, 2, 3]:
        dyn = parse_modes(runs_root, prefix, seed, "wae_dynamic")
        sta = parse_modes(runs_root, prefix, seed, "wae_static_cheap")
        fio = parse_modes(runs_root, prefix, seed, "wae_dynamic_control_forced_io_general")
        prem = parse_modes(runs_root, prefix, seed, "wae_static_premium")
        if not dyn or not sta:
            continue

        dyn_mb = idx_by_query_hash(dyn, "mbpp_eval")
        sta_mb = idx_by_query_hash(sta, "mbpp_eval")
        fio_mb = idx_by_query_hash(fio, "mbpp_eval")
        common = sorted(set(dyn_mb.keys()) & set(sta_mb.keys()))
        conf = {"both_pass": 0, "both_fail": 0, "dynamic_only_pass": 0, "static_only_pass": 0}
        hard_cases: List[Dict[str, object]] = []
        forced_diff = []

        for qh in common:
            d = int(dyn_mb[qh].get("pass", 0))
            s = int(sta_mb[qh].get("pass", 0))
            if d == 1 and s == 1:
                conf["both_pass"] += 1
            elif d == 0 and s == 0:
                conf["both_fail"] += 1
            elif d == 1 and s == 0:
                conf["dynamic_only_pass"] += 1
            else:
                conf["static_only_pass"] += 1
                row = dyn_mb[qh]
                hard_cases.append(
                    {
                        "query_hash": qh,
                        "chosen_workflow_id": row.get("chosen_workflow_id"),
                        "base_endpoint_name": row.get("base_endpoint_name"),
                        "model_id": row.get("model_id"),
                        "endpoint_calls": row.get("endpoint_calls"),
                        "max_tokens_auto_reduce_retries": row.get("max_tokens_auto_reduce_retries"),
                        "sample_cost": row.get("sample_cost"),
                        "sample_latency_s": row.get("sample_latency_s"),
                    }
                )
            if qh in fio_mb:
                drow = dyn_mb[qh]
                frow = fio_mb[qh]
                same_path = (
                    str(drow.get("chosen_workflow_id")) == str(frow.get("chosen_workflow_id"))
                    and int(drow.get("endpoint_calls", 0)) == int(frow.get("endpoint_calls", 0))
                    and int(drow.get("max_tokens_auto_reduce_retries", 0))
                    == int(frow.get("max_tokens_auto_reduce_retries", 0))
                )
                if not same_path:
                    forced_diff.append(
                        {
                            "query_hash": qh,
                            "dynamic_workflow": drow.get("chosen_workflow_id"),
                            "forced_workflow": frow.get("chosen_workflow_id"),
                            "dynamic_calls": drow.get("endpoint_calls"),
                            "forced_calls": frow.get("endpoint_calls"),
                            "dynamic_retry": drow.get("max_tokens_auto_reduce_retries"),
                            "forced_retry": frow.get("max_tokens_auto_reduce_retries"),
                        }
                    )

        hard_cases = sorted(
            hard_cases,
            key=lambda x: (int(x.get("max_tokens_auto_reduce_retries", 0)), float(x.get("sample_cost", 0.0))),
            reverse=True,
        )[: int(args.topk)]
        forced_diff = forced_diff[: int(args.topk)]

        oracle_mbpp = oracle_from_static(sta, prem, "mbpp_eval") if prem else {}
        oracle_he = oracle_from_static(sta, prem, "humaneval_eval") if prem else {}
        if oracle_mbpp:
            agg_oracle_mbpp.append(float(oracle_mbpp.get("oracle_gain_vs_cheap", 0.0)))
        if oracle_he:
            agg_oracle_he.append(float(oracle_he.get("oracle_gain_vs_cheap", 0.0)))

        for k in agg_conf:
            agg_conf[k] += int(conf[k])

        report["seeds"][f"s{seed}"] = {
            "mbpp_confusion_dynamic_vs_static_cheap": conf,
            "dynamic_fail_static_pass_topk": hard_cases,
            "dynamic_selection_mbpp": summarize_selection(dyn, "mbpp_eval"),
            "dynamic_selection_humaneval": summarize_selection(dyn, "humaneval_eval"),
            "dynamic_vs_forced_io_path_diff_topk": forced_diff,
            "oracle_headroom_mbpp": oracle_mbpp,
            "oracle_headroom_humaneval": oracle_he,
        }

    def _avg(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    report["aggregate"] = {
        "mbpp_confusion_dynamic_vs_static_cheap": agg_conf,
        "oracle_gain_vs_cheap_avg_mbpp": _avg(agg_oracle_mbpp),
        "oracle_gain_vs_cheap_avg_humaneval": _avg(agg_oracle_he),
    }

    out_json = Path(f"{out_prefix}.json")
    out_md = Path(f"{out_prefix}.md")
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Round P0 Analysis", ""]
    lines.append(f"- prefix: `{prefix}`")
    lines.append("")
    lines.append("## Aggregate")
    lines.append(
        f"- MBPP confusion (dynamic vs static_cheap): `{report['aggregate']['mbpp_confusion_dynamic_vs_static_cheap']}`"
    )
    lines.append(
        f"- Oracle gain vs cheap (avg): MBPP `{report['aggregate']['oracle_gain_vs_cheap_avg_mbpp']:+.4f}`, "
        f"HumanEval `{report['aggregate']['oracle_gain_vs_cheap_avg_humaneval']:+.4f}`"
    )
    for seed, payload in sorted(report["seeds"].items()):
        lines.append("")
        lines.append(f"## {seed}")
        lines.append(
            f"- MBPP confusion: `{payload['mbpp_confusion_dynamic_vs_static_cheap']}`"
        )
        lines.append(
            f"- Dynamic selection (MBPP): `{payload['dynamic_selection_mbpp']}`"
        )
        lines.append(
            f"- Dynamic selection (HumanEval): `{payload['dynamic_selection_humaneval']}`"
        )
        lines.append(
            f"- Oracle MBPP: `{payload['oracle_headroom_mbpp']}`"
        )
        lines.append(
            f"- Oracle HumanEval: `{payload['oracle_headroom_humaneval']}`"
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(str(out_json))
    print(str(out_md))


if __name__ == "__main__":
    main()
