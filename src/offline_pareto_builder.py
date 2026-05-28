"""Offline Pareto library over candidate workflows, conditioned on role.

Before training the router, we calibrate every candidate workflow on a
shared mini-set drawn from MBPP and/or HumanEval and measure its
(`pass_rate`, `avg_cost`, `avg_latency_s`) triple. Workflows that are
strictly dominated on those three axes are dropped; the rest become the
*Pareto front*.

We then take that Pareto front and split it by which **MAR role** a
workflow is allowed to serve (see `CODE_ROLES` in `workflow_profile.py`).
The output is a JSON file with three things:

- `workflow_metrics`     : the raw (pass_rate, cost, latency) for every
                           candidate, plus its `budget_tier` and
                           `allowed_roles`.
- `pareto_front`         : the ids of the non-dominated workflows.
- `role_pareto_library`  : `{role_name -> [pareto-eligible workflow ids
                           for that role]}`. If no Pareto workflow is
                           allowed for a role, the full Pareto front is
                           returned as a safe fallback.

`run_pilot.py` consumes this JSON as `pareto_library.json` to constrain
the router to a sane operating frontier before training.

The calibration sampler supports `mbpp`, `humaneval`, or `mixed` (50/50
deterministic shuffle) — see GLOSSARY for benchmark notes.
"""

from __future__ import annotations

import json
import os
import re
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

from datasets import load_dataset

from MAR.Utils.globals import Cost

from .safe_exec import run_python_tests
from .workflow_llm import WorkflowLLM
from .workflow_profile import CODE_ROLES
from .workflow_router_patch import EndpointManager


def _mbpp_prompt(item: Dict[str, object]) -> str:
    tests = item["test_list"]
    tests_blob = "\n".join(tests)
    return (
        "**Task**:\n"
        "```python\n"
        f"{item['prompt']}\n"
        "```\n"
        "Your code should pass these tests:\n"
        "```python\n"
        f"{tests_blob}\n"
        "```"
    )


def _humaneval_prompt(item: Dict[str, object]) -> str:
    tests = _humaneval_test(item["test"], item["entry_point"])
    return (
        f"{item['prompt']}\n\n"
        "Your code should pass these tests:\n"
        "```python\n"
        f"{tests}\n"
        "```"
    )


def _humaneval_test(test_code: str, entry_point: str) -> str:
    trailer = f"\n\ncheck({entry_point})\n"
    if f"check({entry_point})" in test_code:
        return test_code
    return test_code + trailer


def _extract_python_block(text: str) -> str:
    match = re.search(r"```python(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


@dataclass
class CalibrationSample:
    prompt: str
    tests: List[str]
    source: str


def _is_passed(output: str, tests: List[str], exec_timeout_s: int) -> int:
    code = _extract_python_block(output)
    try:
        passed, _ = run_python_tests(code, tests, timeout_s=exec_timeout_s)
        return int(bool(passed))
    except Exception:
        return 0


def _is_dominated(a: Dict[str, float], b: Dict[str, float]) -> bool:
    """Return True if a is dominated by b."""
    no_worse = (
        b["pass_rate"] >= a["pass_rate"]
        and b["avg_cost"] <= a["avg_cost"]
        and b["avg_latency_s"] <= a["avg_latency_s"]
    )
    strictly_better = (
        b["pass_rate"] > a["pass_rate"]
        or b["avg_cost"] < a["avg_cost"]
        or b["avg_latency_s"] < a["avg_latency_s"]
    )
    return no_worse and strictly_better


def pareto_front(records: List[Dict[str, float]]) -> List[Dict[str, float]]:
    out = []
    for i, r in enumerate(records):
        dominated = False
        for j, other in enumerate(records):
            if i == j:
                continue
            if _is_dominated(r, other):
                dominated = True
                break
        if not dominated:
            out.append(r)
    return out


def _load_calibration_samples(calibration_size: int, calibration_mix: str, seed: int) -> List[CalibrationSample]:
    mix = calibration_mix.lower().strip()
    if mix == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
        return [
            CalibrationSample(
                prompt=_mbpp_prompt(ds[i]),
                tests=list(ds[i]["test_list"]),
                source="mbpp",
            )
            for i in range(min(calibration_size, len(ds)))
        ]
    if mix == "humaneval":
        ds = load_dataset("openai_humaneval", split="test")
        return [
            CalibrationSample(
                prompt=_humaneval_prompt(ds[i]),
                tests=[_humaneval_test(ds[i]["test"], ds[i]["entry_point"])],
                source="humaneval",
            )
            for i in range(min(calibration_size, len(ds)))
        ]
    if mix == "mixed":
        half = max(calibration_size // 2, 1)
        mbpp = _load_calibration_samples(half, "mbpp", seed)
        humaneval = _load_calibration_samples(calibration_size - half, "humaneval", seed)
        combined = mbpp + humaneval
        rng = random.Random(seed)
        rng.shuffle(combined)
        return combined
    raise ValueError(f"Unknown calibration_mix: {calibration_mix}")


def evaluate_workflow(
    endpoint_manager: EndpointManager,
    workflow_def: Dict[str, object],
    calibration_data: List[CalibrationSample],
    exec_timeout_s: int,
) -> Dict[str, float]:
    wf = WorkflowLLM(
        workflow_id=str(workflow_def["id"]),
        workflow_def=workflow_def,
        endpoint_resolver=endpoint_manager.get,
    )
    solved = 0
    cost_sum = 0.0
    latencies = []
    prev_phase = os.environ.get("WAE_PREMIUM_PHASE")
    prev_split = os.environ.get("WAE_PREMIUM_SPLIT")
    try:
        os.environ["WAE_PREMIUM_PHASE"] = "calibration"
        for item in calibration_data:
            os.environ["WAE_PREMIUM_SPLIT"] = f"calibration_{item.source}"
            prompt = item.prompt
            before = Cost.instance().value
            start = time.time()
            out = wf.gen(prompt)
            elapsed = time.time() - start
            after = Cost.instance().value
            solved += _is_passed(str(out), item.tests, exec_timeout_s=exec_timeout_s)
            cost_sum += (after - before)
            latencies.append(elapsed)
    finally:
        if prev_phase is None:
            os.environ.pop("WAE_PREMIUM_PHASE", None)
        else:
            os.environ["WAE_PREMIUM_PHASE"] = prev_phase
        if prev_split is None:
            os.environ.pop("WAE_PREMIUM_SPLIT", None)
        else:
            os.environ["WAE_PREMIUM_SPLIT"] = prev_split
    n = max(len(calibration_data), 1)
    return {
        "id": str(workflow_def["id"]),
        "pass_rate": solved / n,
        "avg_cost": cost_sum / n,
        "avg_latency_s": sum(latencies) / n if latencies else 0.0,
    }


def build_role_conditioned_library(
    endpoint_manager: EndpointManager,
    workflow_candidates: List[Dict[str, object]],
    calibration_size: int,
    output_path: str,
    exec_timeout_s: int = 15,
    progress_hook: Optional[Callable[[str], None]] = None,
    calibration_mix: str = "mbpp",
    seed: int = 1234,
) -> Tuple[List[Dict[str, float]], Dict[str, List[str]]]:
    calibration_data = _load_calibration_samples(
        calibration_size=calibration_size,
        calibration_mix=calibration_mix,
        seed=seed,
    )

    metrics = []
    total = len(workflow_candidates)
    for i, wf in enumerate(workflow_candidates):
        if progress_hook:
            progress_hook(f"offline_pareto {i+1}/{total} workflow={wf['id']}")
        rec = evaluate_workflow(
            endpoint_manager,
            wf,
            calibration_data,
            exec_timeout_s=exec_timeout_s,
        )
        rec["budget_tier"] = wf["budget_tier"]
        rec["allowed_roles"] = wf["allowed_roles"]
        metrics.append(rec)

    front = pareto_front(metrics)
    role_library: Dict[str, List[str]] = {}
    for role in CODE_ROLES:
        role_library[role] = [
            rec["id"]
            for rec in front
            if role in rec.get("allowed_roles", [])
        ]
        if not role_library[role]:
            role_library[role] = [rec["id"] for rec in front]

    payload = {
        "created_at": datetime.utcnow().isoformat(),
        "calibration_size": calibration_size,
        "calibration_mix": calibration_mix,
        "calibration_sources": {
            "mbpp": sum(1 for x in calibration_data if x.source == "mbpp"),
            "humaneval": sum(1 for x in calibration_data if x.source == "humaneval"),
        },
        "workflow_metrics": metrics,
        "pareto_front": [rec["id"] for rec in front],
        "role_pareto_library": role_library,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return metrics, role_library
