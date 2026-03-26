"""Compare pilot runs and emit iso-cost summary + plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def load_summary(run_dir: str) -> Dict:
    path = Path(run_dir) / "metrics" / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def metric(s: Dict, key: str) -> Dict[str, float]:
    d = s[key]
    return {
        "acc": float(d["accuracy_or_pass1"]),
        "cost": float(d["avg_cost"]),
        "lat_p50": float(d["latency_p50_s"]),
        "lat_p95": float(d["latency_p95_s"]),
        "router_overhead_p50": float(d.get("router_overhead_p50_s", 0.0)),
        "llm_infer_p50": float(d.get("llm_infer_p50_s", 0.0)),
        "test_exec_p50": float(d.get("test_exec_p50_s", 0.0)),
        "calls": int(d["call_count_est"]),
    }


def _safe_rel_gap(a: float, b: float) -> float:
    denom = max(abs(a), 1e-12)
    return abs(a - b) / denom


def _interpolate_acc(
    target_cost: float, points: List[Tuple[str, Dict[str, float]]]
) -> Optional[Dict[str, object]]:
    uniq = {}
    for name, pt in points:
        uniq[float(pt["cost"])] = {"name": name, "acc": float(pt["acc"])}
    pairs = sorted(uniq.items(), key=lambda x: x[0])
    if len(pairs) < 2:
        return None
    min_cost, max_cost = pairs[0][0], pairs[-1][0]
    if target_cost < min_cost or target_cost > max_cost:
        return None

    for i in range(len(pairs) - 1):
        c0, p0 = pairs[i]
        c1, p1 = pairs[i + 1]
        if c0 <= target_cost <= c1:
            if abs(c1 - c0) < 1e-12:
                est = p0["acc"]
            else:
                w = (target_cost - c0) / (c1 - c0)
                est = p0["acc"] + (p1["acc"] - p0["acc"]) * w
            return {
                "estimated_acc": float(est),
                "lower": {"name": p0["name"], "cost": c0, "acc": p0["acc"]},
                "upper": {"name": p1["name"], "cost": c1, "acc": p1["acc"]},
            }
    return None


def pareto_envelope_cost_acc(points: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Return non-dominated baseline points on (cost, acc).

    Objective: minimize cost, maximize accuracy.
    """
    # Deduplicate by cost first, keeping best acc at that cost.
    by_cost: Dict[float, Tuple[str, Dict[str, float]]] = {}
    for name, pt in points.items():
        c = float(pt["cost"])
        a = float(pt["acc"])
        if c not in by_cost or a > float(by_cost[c][1]["acc"]):
            by_cost[c] = (name, pt)

    ordered = [by_cost[c] for c in sorted(by_cost.keys())]
    envelope: Dict[str, Dict[str, float]] = {}
    best_acc = float("-inf")
    for name, pt in ordered:
        acc = float(pt["acc"])
        if acc > best_acc + 1e-12:
            envelope[name] = pt
            best_acc = acc
    return envelope


def iso_cost_analysis(
    dynamic: Dict[str, float],
    baselines: Dict[str, Dict[str, float]],
    tolerance: float = 0.05,
    allow_interpolation: bool = True,
) -> Dict[str, object]:
    envelope = pareto_envelope_cost_acc(baselines)
    d_cost = float(dynamic["cost"])
    in_tol = {
        name: pt
        for name, pt in envelope.items()
        if _safe_rel_gap(float(pt["cost"]), d_cost) <= tolerance
    }

    if in_tol:
        ref_name, ref_pt = max(in_tol.items(), key=lambda kv: kv[1]["acc"])
        baseline_acc = float(ref_pt["acc"])
        delta = float(dynamic["acc"]) - baseline_acc
        return {
            "method": "tolerance_band",
            "comparable": True,
            "tolerance": tolerance,
            "envelope_size": len(envelope),
            "candidate_baselines": sorted(in_tol.keys()),
            "reference": {
                "type": "baseline",
                "name": ref_name,
                "cost": float(ref_pt["cost"]),
                "acc": baseline_acc,
            },
            "delta_acc": delta,
            "pass_0.03": bool(delta >= 0.03),
        }

    if allow_interpolation:
        interp = _interpolate_acc(d_cost, list(envelope.items()))
        if interp is not None:
            baseline_acc = float(interp["estimated_acc"])
            delta = float(dynamic["acc"]) - baseline_acc
            return {
                "method": "linear_interpolation",
                "comparable": True,
                "tolerance": tolerance,
                "envelope_size": len(envelope),
                "candidate_baselines": sorted(envelope.keys()),
                "reference": {
                    "type": "interpolated",
                    "estimated_acc": baseline_acc,
                    "lower": interp["lower"],
                    "upper": interp["upper"],
                },
                "delta_acc": delta,
                "pass_0.03": bool(delta >= 0.03),
            }

    name, near = min(envelope.items(), key=lambda kv: abs(kv[1]["cost"] - d_cost))
    delta = float(dynamic["acc"]) - float(near["acc"])
    return {
        "method": "nearest_fallback",
        "comparable": False,
        "tolerance": tolerance,
        "envelope_size": len(envelope),
        "candidate_baselines": sorted(envelope.keys()),
        "reference": {
            "type": "baseline",
            "name": name,
            "cost": float(near["cost"]),
            "acc": float(near["acc"]),
        },
        "delta_acc": delta,
        "pass_0.03": bool(delta >= 0.03),
    }


def dominance_analysis(dynamic: Dict[str, float], baselines: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    dominated_cost_acc = []
    dominated_cost_acc_lat = []
    for name, b in baselines.items():
        cond_cost_acc = (
            b["cost"] <= dynamic["cost"]
            and b["acc"] >= dynamic["acc"]
            and (b["cost"] < dynamic["cost"] or b["acc"] > dynamic["acc"])
        )
        if cond_cost_acc:
            dominated_cost_acc.append(name)

        cond_all = (
            cond_cost_acc
            and b["lat_p50"] <= dynamic["lat_p50"]
            and (
                b["lat_p50"] < dynamic["lat_p50"]
                or b["cost"] < dynamic["cost"]
                or b["acc"] > dynamic["acc"]
            )
        )
        if cond_all:
            dominated_cost_acc_lat.append(name)

    return {
        "dominated_on_cost_acc": bool(dominated_cost_acc),
        "dominated_by_cost_acc": sorted(dominated_cost_acc),
        "dominated_on_cost_acc_p50lat": bool(dominated_cost_acc_lat),
        "dominated_by_cost_acc_p50lat": sorted(dominated_cost_acc_lat),
    }


def apply_success_rule(
    iso_cost: Dict[str, object], dominance: Dict[str, object]
) -> Dict[str, object]:
    """Dominance-first success rule.

    1) If dominated on either axis set, it is FAIL.
    2) Otherwise require iso-cost comparable and +0.03 acc.
    """
    dominated = bool(dominance["dominated_on_cost_acc"]) or bool(
        dominance["dominated_on_cost_acc_p50lat"]
    )
    if dominated:
        return {
            "pass": False,
            "reason": "dominated_by_baseline",
            "dominance_first": True,
            "requires_iso_cost_check": False,
        }
    if not bool(iso_cost["comparable"]):
        return {
            "pass": False,
            "reason": "iso_cost_not_comparable",
            "dominance_first": True,
            "requires_iso_cost_check": True,
        }
    return {
        "pass": bool(iso_cost["pass_0.03"]),
        "reason": "iso_cost_delta_threshold",
        "dominance_first": True,
        "requires_iso_cost_check": True,
    }


def save_plot(ds_key: str, scores: Dict[str, Dict[str, float]], out_png: Path) -> None:
    colors = {
        "masrouter": "#1f77b4",
        "masrouter_cheap": "#5fa2dd",
        "masrouter_balanced": "#2f7fbf",
        "masrouter_premium": "#0f4f8f",
        "wae_dynamic": "#ff7f0e",
        "wae_dynamic_no_premium": "#9467bd",
        "wae_dynamic_prior_gated": "#8c564b",
        "wae_dynamic_roi_gated": "#8c564b",
        "wae_dynamic_hardcase_gate": "#bcbd22",
        "wae_cheap_first_escalate": "#17becf",
        "wae_dynamic_control_forced_io": "#7f7f7f",
        "wae_static_cheap": "#2ca02c",
        "wae_static_premium": "#d62728",
    }
    fig, ax = plt.subplots(figsize=(7, 5))
    for mode, pt in scores.items():
        x = pt["cost"]
        y = pt["acc"]
        ax.scatter([x], [y], s=90, color=colors.get(mode, "black"))
        ax.annotate(mode, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel("Average Cost")
    ax.set_ylabel("Accuracy / pass@1")
    ax.set_title(f"Cost-Accuracy: {ds_key}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--masrouter_run", required=True, type=str)
    p.add_argument("--wae_dynamic_run", required=True, type=str)
    p.add_argument("--wae_static_cheap_run", required=True, type=str)
    p.add_argument("--wae_static_premium_run", required=True, type=str)
    p.add_argument("--wae_dynamic_no_premium_run", default=None, type=str)
    p.add_argument("--wae_dynamic_prior_gated_run", default=None, type=str)
    p.add_argument("--wae_dynamic_roi_gated_run", default=None, type=str)
    p.add_argument("--wae_dynamic_hardcase_gate_run", default=None, type=str)
    p.add_argument("--wae_cheap_first_escalate_run", default=None, type=str)
    p.add_argument("--wae_dynamic_control_forced_io_run", default=None, type=str)
    p.add_argument(
        "--masrouter_curve_runs",
        default="",
        type=str,
        help="Comma-separated label=run_dir for extra masrouter operating points.",
    )
    p.add_argument(
        "--target_mode",
        default="wae_dynamic",
        type=str,
        help="Mode name to evaluate against baselines (default: wae_dynamic).",
    )
    p.add_argument("--out_prefix", default="/workspace/wae_router_pilot/runs/pilot6h_compare", type=str)
    p.add_argument("--iso_tolerance", default=0.05, type=float)
    p.add_argument("--disable_interpolation", action="store_true")
    p.add_argument(
        "--budget_points",
        default="0.00026,0.00030,0.00040",
        type=str,
        help="Comma-separated budget cutoffs used for best-under-budget analysis.",
    )
    args = p.parse_args()

    run_dirs = {
        "masrouter": args.masrouter_run,
        "wae_dynamic": args.wae_dynamic_run,
        "wae_static_cheap": args.wae_static_cheap_run,
        "wae_static_premium": args.wae_static_premium_run,
    }
    if args.wae_dynamic_no_premium_run:
        run_dirs["wae_dynamic_no_premium"] = args.wae_dynamic_no_premium_run
    if args.wae_dynamic_prior_gated_run:
        run_dirs["wae_dynamic_prior_gated"] = args.wae_dynamic_prior_gated_run
    if args.wae_dynamic_roi_gated_run:
        run_dirs["wae_dynamic_roi_gated"] = args.wae_dynamic_roi_gated_run
    if args.wae_dynamic_hardcase_gate_run:
        run_dirs["wae_dynamic_hardcase_gate"] = args.wae_dynamic_hardcase_gate_run
    if args.wae_cheap_first_escalate_run:
        run_dirs["wae_cheap_first_escalate"] = args.wae_cheap_first_escalate_run
    if args.wae_dynamic_control_forced_io_run:
        run_dirs["wae_dynamic_control_forced_io"] = args.wae_dynamic_control_forced_io_run
    if args.masrouter_curve_runs.strip():
        for item in args.masrouter_curve_runs.split(","):
            token = item.strip()
            if not token:
                continue
            if "=" not in token:
                raise ValueError(
                    f"Invalid masrouter curve token '{token}'. Expected label=run_dir."
                )
            label, run_dir = token.split("=", 1)
            label = label.strip()
            run_dir = run_dir.strip()
            if not label or not run_dir:
                raise ValueError(
                    f"Invalid masrouter curve token '{token}'. Expected label=run_dir."
                )
            run_dirs[label] = run_dir
    summaries = {k: load_summary(v) for k, v in run_dirs.items()}

    if args.target_mode not in run_dirs:
        raise ValueError(
            f"--target_mode '{args.target_mode}' is not present in provided run dirs: {sorted(run_dirs.keys())}"
        )

    out = {
        "runs": run_dirs,
        "datasets": {},
        "iso_cost": {},
        "dominance": {},
        "verdict": {},
        "budget_analysis": {},
        "criterion": "dominance-first then +0.03 acc at iso-cost",
        "settings": {
            "iso_tolerance": float(args.iso_tolerance),
            "allow_interpolation": not bool(args.disable_interpolation),
            "budget_points": args.budget_points,
        },
    }
    budgets: List[float] = []
    for tok in str(args.budget_points).split(","):
        tok = tok.strip()
        if not tok:
            continue
        budgets.append(float(tok))

    for ds in ["mbpp_eval", "humaneval_eval"]:
        scores = {k: metric(v, ds) for k, v in summaries.items()}
        out["datasets"][ds] = scores
        baselines = {k: v for k, v in scores.items() if k != args.target_mode}
        out["iso_cost"][ds] = iso_cost_analysis(
            dynamic=scores[args.target_mode],
            baselines=baselines,
            tolerance=float(args.iso_tolerance),
            allow_interpolation=not bool(args.disable_interpolation),
        )
        out["dominance"][ds] = dominance_analysis(scores[args.target_mode], baselines)
        out["verdict"][ds] = apply_success_rule(
            iso_cost=out["iso_cost"][ds],
            dominance=out["dominance"][ds],
        )
        budget_rows = []
        for b in budgets:
            feasible = {
                mode: v for mode, v in scores.items() if float(v["cost"]) <= float(b)
            }
            if feasible:
                best_mode, best_pt = max(
                    feasible.items(), key=lambda kv: float(kv[1]["acc"])
                )
                row = {
                    "budget": float(b),
                    "best_mode": best_mode,
                    "best_acc": float(best_pt["acc"]),
                    "num_feasible": int(len(feasible)),
                    "feasible_acc_by_mode": {
                        k: float(v["acc"]) for k, v in sorted(feasible.items())
                    },
                }
            else:
                row = {
                    "budget": float(b),
                    "best_mode": None,
                    "best_acc": None,
                    "num_feasible": 0,
                    "feasible_acc_by_mode": {},
                }
            budget_rows.append(row)
        out["budget_analysis"][ds] = budget_rows
        save_plot(ds, scores, Path(f"{args.out_prefix}_cost_acc_{ds}.png"))

    json_path = Path(f"{args.out_prefix}.json")
    md_path = Path(f"{args.out_prefix}.md")
    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Run Comparison")
    lines.append("")
    for ds in ["mbpp_eval", "humaneval_eval"]:
        lines.append(f"## {ds}")
        lines.append("| mode | acc | avg_cost | p50 | p95 | router_p50 | llm_p50 | test_p50 | calls |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        ordered_modes = [
            "masrouter",
            "masrouter_cheap",
            "masrouter_balanced",
            "masrouter_premium",
            "wae_static_cheap",
            "wae_static_premium",
            "wae_dynamic_no_premium",
            "wae_dynamic_prior_gated",
            "wae_dynamic_roi_gated",
            "wae_dynamic_hardcase_gate",
            "wae_dynamic_control_forced_io",
            "wae_cheap_first_escalate",
            "wae_dynamic",
        ]
        for mode in sorted(out["datasets"][ds].keys()):
            if mode.startswith("masrouter_") and mode not in ordered_modes:
                ordered_modes.insert(1, mode)
        for mode in ordered_modes:
            if mode not in out["datasets"][ds]:
                continue
            d = out["datasets"][ds][mode]
            lines.append(
                f"| {mode} | {d['acc']:.4f} | {d['cost']:.8f} | {d['lat_p50']:.3f} | {d['lat_p95']:.3f} | "
                f"{d.get('router_overhead_p50', 0.0):.3f} | {d.get('llm_infer_p50', 0.0):.3f} | {d.get('test_exec_p50', 0.0):.3f} | {d['calls']} |"
            )
        ic = out["iso_cost"][ds]
        dom = out["dominance"][ds]
        vd = out["verdict"][ds]
        lines.append("")
        lines.append(f"- iso-cost method: `{ic['method']}` (comparable=`{ic['comparable']}`)")
        lines.append(f"- target mode: `{args.target_mode}`")
        lines.append(f"- baseline envelope size: `{ic.get('envelope_size', 0)}`")
        lines.append(f"- delta acc ({args.target_mode} - reference): `{ic['delta_acc']:+.4f}`")
        lines.append(f"- success(+0.03): `{ic['pass_0.03']}`")
        lines.append(
            f"- dominated on (cost, acc): `{dom['dominated_on_cost_acc']}` by `{dom['dominated_by_cost_acc']}`"
        )
        lines.append(
            f"- dominated on (cost, acc, p50lat): `{dom['dominated_on_cost_acc_p50lat']}` by `{dom['dominated_by_cost_acc_p50lat']}`"
        )
        lines.append(f"- final verdict (dominance-first): `{vd['pass']}` (`{vd['reason']}`)")
        lines.append(f"- plot: `{args.out_prefix}_cost_acc_{ds}.png`")
        lines.append("- budget best-under-budget:")
        for row in out["budget_analysis"][ds]:
            lines.append(
                f"  - B={row['budget']:.8f}: best=`{row['best_mode']}` acc=`{row['best_acc']}` feasible={row['num_feasible']}"
            )
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
