"""Aggregate pilot run outputs and compute iso-cost improvement signal."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List


def latest_summary_for_mode(runs_root: str, mode: str) -> str:
    pattern = os.path.join(runs_root, f"{mode}_*", "metrics", "summary.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No summary found for mode={mode}")
    return files[-1]


def load_summary(path: str) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def score(summary: Dict, dataset_key: str) -> Dict[str, float]:
    d = summary[dataset_key]
    return {
        "acc": float(d["accuracy_or_pass1"]),
        "cost": float(d["avg_cost"]),
        "lat_p50": float(d["latency_p50_s"]),
        "lat_p95": float(d["latency_p95_s"]),
    }


def nearest_iso_cost_delta(target: Dict[str, float], cands: List[Dict[str, float]]) -> float:
    best = min(cands, key=lambda x: abs(x["cost"] - target["cost"]))
    return target["acc"] - best["acc"]


def main() -> None:
    p = argparse.ArgumentParser()
    _wae_root = os.environ.get("WAE_ROUTER_PILOT_ROOT", "/workspace/wae_router_pilot")
    p.add_argument(
        "--runs_root",
        type=str,
        default=os.environ.get("WAE_RUNS_ROOT", f"{_wae_root}/runs"),
    )
    p.add_argument(
        "--output",
        type=str,
        default=os.environ.get("WAE_ANALYSIS_OUTPUT", f"{_wae_root}/runs/analysis_latest.json"),
    )
    args = p.parse_args()

    modes = ["masrouter", "wae_dynamic", "wae_static_cheap", "wae_static_premium"]
    summaries = {m: load_summary(latest_summary_for_mode(args.runs_root, m)) for m in modes}

    out = {"modes": {}, "iso_cost_checks": {}}
    for dataset_key in ["mbpp_eval", "humaneval_eval"]:
        mode_scores = {m: score(summaries[m], dataset_key) for m in modes}
        out["modes"][dataset_key] = mode_scores
        delta_vs_baselines = nearest_iso_cost_delta(
            mode_scores["wae_dynamic"],
            [
                mode_scores["masrouter"],
                mode_scores["wae_static_cheap"],
                mode_scores["wae_static_premium"],
            ],
        )
        out["iso_cost_checks"][dataset_key] = {
            "delta_accuracy": delta_vs_baselines,
            "pass_threshold_0.03": bool(delta_vs_baselines >= 0.03),
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

