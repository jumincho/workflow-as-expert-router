"""Main experiment runner for the WaE-Router pilot.

End-to-end orchestration of one routing mode against one set of evaluation
benchmarks. A single invocation:

1. Boots `EndpointManager` against `config/model_endpoints*.yaml`, warms
   each declared vLLM endpoint, and (optionally) refuses to proceed if
   the endpoint set is not heterogeneous (>=2 distinct model ids /
   base URLs).
2. Installs `LLMRuntimePatch` so MasRouter's `LLMRegistry.get` resolves
   `wf::<id>` entries to `WorkflowLLM` and bare endpoint names to
   `EndpointLLM`.
3. Loads MBPP and HumanEval splits (train + eval), runs the offline
   role-conditioned Pareto calibration (`offline_pareto_builder`), and
   trains / re-uses a `WaERouter` according to the `--mode` flag:
     - `masrouter`                 — stock model-level routing baseline.
     - `wae_dynamic`               — workflow-aware router.
     - `wae_static_cheap`          — force the cheap workflow everywhere.
     - `wae_static_premium`        — force the premium workflow everywhere.
     - `wae_dynamic_*` (no_premium, prior_gated, roi_gated, hardcase_gate,
       cheap_first_escalate)       — ablations of the dynamic policy.
4. Evaluates on MBPP and HumanEval, recording per-sample passes,
   latency breakdowns (router overhead / LLM inference / test exec),
   prompt and output hashes, retry counters, and the chosen workflow id.
5. Writes `runs/<run_id>/metrics/summary.json`, `report.md`, sample
   traces under `logs/`, and a heartbeat status file consumed by
   `monitor.PilotMonitor`. Companion outputs (compare reports, plots,
   round-level P0 diagnostics) are produced by `compare_runs.py` and
   `analyze_round_p0.py` against the directories this runner produces.

The runner depends on the upstream `MAR` (MasRouter) package which is
*not* vendored; the `MASROUTER_PATH` env var points to a separate
checkout (default `/workspace/masrouter`). All output paths default
to `WAE_ROUTER_PILOT_ROOT/runs/` (override with `WAE_RUNS_ROOT`).

See GLOSSARY.md for the env vars, the mode labels, and the snapshot
formats consumed by `monitor.py` and `compare_runs.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import signal
import sys
import time
import traceback
from contextlib import contextmanager
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from loguru import logger

# WaE-Router consumes the upstream MasRouter (MAR) package without
# vendoring it. Default to /workspace/masrouter for the original
# author's host layout; override with the MASROUTER_PATH environment
# variable on any other machine.
_masrouter_path = os.environ.get("MASROUTER_PATH", "/workspace/masrouter")
if _masrouter_path not in sys.path:
    sys.path.append(_masrouter_path)

from MAR.MasRouter.mas_router import MasRouter
from MAR.Utils.globals import CompletionTokens, Cost, PromptTokens
from MAR.Utils.utils import fix_random_seed
from MAR.LLM import llm_embedding as mar_llm_embedding

from .monitor import PilotMonitor
from .offline_pareto_builder import build_role_conditioned_library
from .safe_exec import run_python_tests
from .workflow_llm import reset_runtime_telemetry
from .workflow_router_patch import EndpointManager, LLMRuntimePatch, WaERouter, default_workflow_candidates


@dataclass
class Sample:
    query: str
    tests: List[str]
    source: str


TASKS_PROFILE = [
    {
        "Name": "Math",
        "Description": "A mathematics problem often involves logical reasoning and calculations.",
    },
    {
        "Name": "Commonsense",
        "Description": "A commonsense question involving general world knowledge and practical reasoning.",
    },
    {
        "Name": "Code",
        "Description": "A coding task requiring code generation, debugging, or algorithm design.",
    },
]


REASONING_PROFILE = [
    {"Name": "IO", "Description": "Single-agent direct output."},
    {"Name": "CoT", "Description": "Single-agent chain-of-thought reasoning."},
    {"Name": "Chain", "Description": "Multi-agent chain reasoning."},
    {"Name": "FullConnected", "Description": "Multi-agent full graph reasoning."},
    {"Name": "Debate", "Description": "Multi-agent debate reasoning."},
    {"Name": "Reflection", "Description": "Reflective iterative reasoning."},
]


def parse_args() -> Tuple[argparse.Namespace, argparse.ArgumentParser]:
    p = argparse.ArgumentParser(description="WaE-Router pilot runner")
    p.add_argument(
        "--mode",
        required=True,
        choices=[
            "masrouter",
            "wae_dynamic",
            "wae_dynamic_hardcase_gate",
            "wae_dynamic_prior_gated",
            "wae_dynamic_roi_gated",
            "wae_dynamic_no_premium",
            "wae_static_cheap",
            "wae_static_premium",
            "wae_cheap_first_escalate",
        ],
    )
    p.add_argument("--train_samples", type=int, default=64)
    p.add_argument("--test_samples_mbpp", type=int, default=80)
    p.add_argument("--test_samples_humaneval", type=int, default=80)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--cost_rate", type=float, default=300.0)
    p.add_argument("--max_agent", type=int, default=4)
    p.add_argument("--calibration_size", type=int, default=24)
    p.add_argument("--calibration_mix", type=str, default="mbpp", choices=["mbpp", "humaneval", "mixed"])
    p.add_argument("--exec_timeout_s", type=int, default=15)
    p.add_argument(
        "--batch_timeout_s",
        type=int,
        default=180,
        help="Hard timeout per batch forward/eval step to prevent hangs.",
    )
    p.add_argument("--force_workflow_id", type=str, default=None)
    p.add_argument("--cheap_workflow_id", type=str, default="wf_io_general")
    p.add_argument("--premium_workflow_id", type=str, default="wf_gen3_test_select_general")
    p.add_argument(
        "--exclude_budget_tiers",
        type=str,
        default="",
        help="Comma-separated workflow budget tiers to exclude (e.g., premium).",
    )
    p.add_argument(
        "--model_endpoints",
        type=str,
        default=os.environ.get(
            "WAE_MODEL_ENDPOINTS",
            f"{os.environ.get('WAE_ROUTER_PILOT_ROOT', '/workspace/wae_router_pilot')}/config/model_endpoints.yaml",
        ),
    )
    p.add_argument(
        "--experiment_config",
        type=str,
        default=os.environ.get(
            "WAE_EXPERIMENT_CONFIG",
            f"{os.environ.get('WAE_ROUTER_PILOT_ROOT', '/workspace/wae_router_pilot')}/config/experiment.yaml",
        ),
    )
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no_fallback", action="store_true")
    p.add_argument("--require_heterogeneous_endpoints", action="store_true")
    p.add_argument("--endpoint_ready_retries", type=int, default=6)
    p.add_argument("--endpoint_ready_interval_s", type=float, default=10.0)
    p.add_argument("--endpoint_warmup", action="store_true")
    p.add_argument(
        "--reuse_pareto_json",
        type=str,
        default="",
        help="If provided, reuse offline Pareto metrics from this JSON when available; otherwise write newly built metrics to this path.",
    )
    p.add_argument("--workflow_prior_beta", type=float, default=0.8)
    p.add_argument("--premium_prior_epsilon", type=float, default=0.02)
    p.add_argument("--enable_sample_trace", action="store_true")
    p.add_argument("--deterministic_inference", action="store_true")
    p.add_argument(
        "--inject_tests_into_humaneval_query",
        action="store_true",
        help="Append executable tests into HumanEval prompt so premium test-select can access them.",
    )
    p.add_argument(
        "--premium_require_tests",
        type=int,
        default=1,
        help="If 1, premium test-select falls back when inline tests are unavailable.",
    )
    p.add_argument(
        "--roi_gate_margin",
        type=float,
        default=0.0,
        help="Require premium ROI >= cheap ROI * (1 + margin).",
    )
    p.add_argument(
        "--workflow_entropy_reg",
        type=float,
        default=0.0,
        help="Entropy regularization coefficient for workflow routing (dynamic modes only).",
    )
    p.add_argument(
        "--hardcase_tau",
        type=float,
        default=0.5,
        help="Fail-probability threshold for wae_dynamic_hardcase_gate.",
    )
    p.add_argument(
        "--hardcase_calibration_size",
        type=int,
        default=64,
        help="Calibration sample count for cheap-fail predictor.",
    )
    p.add_argument(
        "--hardcase_premium_workflow_id",
        type=str,
        default="wf_gen2_test_select_general",
        help="Premium workflow id used by wae_dynamic_hardcase_gate.",
    )
    p.add_argument(
        "--deterministic_router_components",
        action="store_true",
        help="Force deterministic argmax selection for collab/role/workflow in WaE routing.",
    )
    return p.parse_args(), p


def load_exp_config(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def setup_run_dirs(run_id: str) -> Dict[str, str]:
    wae_root = os.environ.get("WAE_ROUTER_PILOT_ROOT", "/workspace/wae_router_pilot")
    root = f"{wae_root}/runs/{run_id}"
    dirs = {
        "root": root,
        "logs": os.path.join(root, "logs"),
        "metrics": os.path.join(root, "metrics"),
        "plots": os.path.join(root, "plots"),
        "checkpoints": os.path.join(root, "checkpoints"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def mbpp_prompt(item: Dict[str, object]) -> str:
    tests_blob = "\n".join(item["test_list"])
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


def extract_python_block(text: str) -> str:
    m = re.search(r"```python(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def short_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def build_humaneval_test(test_code: str, entry_point: str) -> str:
    trailer = f"\n\ncheck({entry_point})\n"
    if f"check({entry_point})" in test_code:
        return test_code
    return test_code + trailer


def load_mbpp_samples(train_n: int, test_n: int) -> Tuple[List[Sample], List[Sample]]:
    ds_train = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
    ds_test = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    train = [
        Sample(query=mbpp_prompt(ds_train[i]), tests=list(ds_train[i]["test_list"]), source="mbpp_train")
        for i in range(min(train_n, len(ds_train)))
    ]
    test = [
        Sample(query=mbpp_prompt(ds_test[i]), tests=list(ds_test[i]["test_list"]), source="mbpp_test")
        for i in range(min(test_n, len(ds_test)))
    ]
    return train, test


def load_humaneval_samples(test_n: int) -> List[Sample]:
    ds = load_dataset("openai_humaneval", split="test")
    data = []
    for i in range(min(test_n, len(ds))):
        item = ds[i]
        tests = [build_humaneval_test(item["test"], item["entry_point"])]
        query = str(item["prompt"])
        data.append(Sample(query=query, tests=tests, source="humaneval_test"))
    return data


def load_humaneval_samples_with_inline_tests(test_n: int) -> List[Sample]:
    ds = load_dataset("openai_humaneval", split="test")
    data = []
    for i in range(min(test_n, len(ds))):
        item = ds[i]
        tests = [build_humaneval_test(item["test"], item["entry_point"])]
        query = (
            f"{item['prompt']}\n\n"
            "Your code should pass these tests:\n"
            "```python\n"
            f"{tests[0]}\n"
            "```"
        )
        data.append(Sample(query=query, tests=tests, source="humaneval_test"))
    return data


def batched(data: List[Sample], batch_size: int) -> List[List[Sample]]:
    return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]


def evaluate_batch_outputs(results: List[str], batch: List[Sample], exec_timeout_s: int) -> List[int]:
    solved = []
    for out, item in zip(results, batch):
        code = extract_python_block(out)
        try:
            ok, _ = run_python_tests(code, item.tests, timeout_s=exec_timeout_s)
            solved.append(int(bool(ok)))
        except Exception:
            solved.append(0)
    return solved


def percentile(arr: List[float], p: float) -> float:
    if not arr:
        return 0.0
    return float(np.percentile(np.array(arr), p))


def phase_name_from_split(train: bool, split_name: str) -> str:
    if train:
        return "train"
    split = str(split_name).strip().lower()
    if split.startswith("calibration"):
        return "calibration"
    if split == "mbpp_eval":
        return "eval_mbpp"
    if split == "humaneval_eval":
        return "eval_humaneval"
    return "eval"


def save_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


class BatchTimeoutError(RuntimeError):
    """Raised when a batch-level timeout is exceeded."""


@contextmanager
def batch_timeout_guard(timeout_s: int):
    if timeout_s <= 0:
        yield
        return
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):
        raise BatchTimeoutError(f"batch timeout exceeded: {timeout_s}s")

    prev_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prev_handler)


def reset_global_counters() -> None:
    Cost.instance().reset()
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()


def difficulty_threshold(data: List[Sample]) -> int:
    if not data:
        return 0
    lens = sorted(len(x.query) for x in data)
    return int(lens[len(lens) // 2])


def hardcase_features(sample: Sample) -> List[float]:
    query = str(sample.query)
    tests_blob = "\n".join(sample.tests) if sample.tests else ""
    assert_cnt = sum(1 for line in tests_blob.splitlines() if line.strip().startswith("assert "))
    def_cnt = len(re.findall(r"\bdef\s+\w+\s*\(", query))
    return [
        float(len(query)),
        float(len(sample.tests)),
        float(assert_cnt),
        float(def_cnt),
    ]


def build_hardcase_calibration_set(
    train_mbpp: List[Sample],
    test_humaneval: List[Sample],
    calibration_size: int,
    mix: str,
    seed: int,
) -> List[Sample]:
    rng = random.Random(int(seed))
    if calibration_size <= 0:
        return []
    mix = str(mix).strip().lower()
    if mix == "mbpp":
        pool = list(train_mbpp)
    elif mix == "humaneval":
        pool = list(test_humaneval)
    else:
        m = max(1, calibration_size // 2)
        h = max(1, calibration_size - m)
        pool = list(train_mbpp[:m]) + list(test_humaneval[:h])
    if len(pool) > calibration_size:
        rng.shuffle(pool)
        pool = pool[:calibration_size]
    return pool


def fit_hardcase_logistic(
    samples: List[Sample],
    cheap_fail_labels: List[int],
    seed: int,
) -> Dict[str, object]:
    if not samples:
        return {
            "feature_names": ["query_len", "num_tests", "num_assert", "num_defs"],
            "coef": [0.0, 0.0, 0.0, 0.0],
            "bias": 0.0,
            "mean": [0.0, 0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0, 1.0],
            "train_loss": 0.0,
            "train_acc": 0.0,
            "label_fail_rate": 0.0,
            "n_samples": 0,
        }

    x_np = np.array([hardcase_features(s) for s in samples], dtype=np.float32)
    y_np = np.array([int(v) for v in cheap_fail_labels], dtype=np.float32).reshape(-1, 1)
    mean = x_np.mean(axis=0)
    std = x_np.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x_norm = (x_np - mean) / std

    torch.manual_seed(int(seed))
    x = torch.tensor(x_norm, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    w = torch.zeros((x.shape[1], 1), dtype=torch.float32, requires_grad=True)
    b = torch.zeros((1,), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.08)

    for _ in range(240):
        logits = x @ w + b
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        probs = torch.sigmoid(x @ w + b).cpu().numpy().reshape(-1)
    preds = (probs >= 0.5).astype(np.int32)
    train_acc = float((preds == y_np.reshape(-1).astype(np.int32)).mean())
    fail_rate = float(y_np.mean()) if len(y_np) > 0 else 0.0

    return {
        "feature_names": ["query_len", "num_tests", "num_assert", "num_defs"],
        "coef": [float(v) for v in w.detach().cpu().reshape(-1).tolist()],
        "bias": float(b.detach().cpu().item()),
        "mean": [float(v) for v in mean.tolist()],
        "std": [float(v) for v in std.tolist()],
        "train_loss": float(loss.detach().cpu().item()),
        "train_acc": float(train_acc),
        "label_fail_rate": float(fail_rate),
        "n_samples": int(len(samples)),
    }


def hardcase_fail_prob(sample: Sample, predictor: Optional[Dict[str, object]]) -> float:
    if not predictor:
        return 0.0
    feat = np.array(hardcase_features(sample), dtype=np.float32)
    mean = np.array(predictor.get("mean", [0.0] * len(feat)), dtype=np.float32)
    std = np.array(predictor.get("std", [1.0] * len(feat)), dtype=np.float32)
    coef = np.array(predictor.get("coef", [0.0] * len(feat)), dtype=np.float32)
    bias = float(predictor.get("bias", 0.0))
    std = np.where(np.abs(std) < 1e-6, 1.0, std)
    x = (feat - mean) / std
    logit = float(np.dot(x, coef) + bias)
    prob = 1.0 / (1.0 + np.exp(-logit))
    return max(0.0, min(1.0, float(prob)))


class RoutingTracker:
    def __init__(self, workflow_candidates: List[Dict[str, object]]):
        self.workflow_to_model = {
            f"wf::{wf['id']}": str(wf.get("base_model", "unknown"))
            for wf in workflow_candidates
        }
        self.by_split: Dict[str, Dict[str, Any]] = {}

    def _split(self, split_name: str) -> Dict[str, Any]:
        if split_name not in self.by_split:
            self.by_split[split_name] = {
                "role_workflow_counts": defaultdict(lambda: defaultdict(int)),
                "workflow_success": defaultdict(lambda: {"count": 0, "solved": 0}),
                "endpoint_usage": defaultdict(int),
                "difficulty": {
                    "easy": {"count": 0, "solved": 0, "cost_sum": 0.0, "latency_sum": 0.0, "workflow_counts": defaultdict(int)},
                    "hard": {"count": 0, "solved": 0, "cost_sum": 0.0, "latency_sum": 0.0, "workflow_counts": defaultdict(int)},
                },
                "case_studies": [],
            }
        return self.by_split[split_name]

    def record(
        self,
        *,
        split_name: str,
        sample: Sample,
        selected_workflows: List[str],
        selected_roles: List[str],
        solved: int,
        sample_cost: float,
        sample_latency_s: float,
        threshold: int,
    ) -> None:
        split = self._split(split_name)
        bucket = "hard" if len(sample.query) > threshold else "easy"

        b = split["difficulty"][bucket]
        b["count"] += 1
        b["solved"] += int(solved)
        b["cost_sum"] += float(sample_cost)
        b["latency_sum"] += float(sample_latency_s)

        for wf in selected_workflows:
            b["workflow_counts"][wf] += 1
            split["workflow_success"][wf]["count"] += 1
            split["workflow_success"][wf]["solved"] += int(solved)
            endpoint = self.workflow_to_model.get(wf, "unknown")
            split["endpoint_usage"][endpoint] += 1

        for role, wf in zip(selected_roles, selected_workflows):
            split["role_workflow_counts"][role][wf] += 1

        if len(split["case_studies"]) < 10:
            qhash = hashlib.md5(sample.query.encode("utf-8")).hexdigest()[:8]
            split["case_studies"].append(
                {
                    "query_hash": qhash,
                    "query_preview": sample.query[:180],
                    "num_tests": len(sample.tests),
                    "bucket": bucket,
                    "selected_roles": selected_roles,
                    "selected_workflows": selected_workflows,
                    "solved": int(solved),
                    "sample_cost": float(sample_cost),
                    "sample_latency_s": float(sample_latency_s),
                }
            )

    def to_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for split_name, split in self.by_split.items():
            diff_out = {}
            for bucket in ["easy", "hard"]:
                b = split["difficulty"][bucket]
                cnt = max(int(b["count"]), 1)
                diff_out[bucket] = {
                    "count": int(b["count"]),
                    "accuracy": float(b["solved"]) / cnt,
                    "avg_cost": float(b["cost_sum"]) / cnt,
                    "avg_latency_s": float(b["latency_sum"]) / cnt,
                    "workflow_counts": dict(sorted(b["workflow_counts"].items())),
                }
            wf_success = {}
            for wf, v in split["workflow_success"].items():
                cnt = max(int(v["count"]), 1)
                wf_success[wf] = {
                    "count": int(v["count"]),
                    "solved": int(v["solved"]),
                    "pass_rate": float(v["solved"]) / cnt,
                }
            out[split_name] = {
                "role_workflow_counts": {
                    role: dict(sorted(wf_counts.items()))
                    for role, wf_counts in split["role_workflow_counts"].items()
                },
                "workflow_success": dict(sorted(wf_success.items())),
                "top_workflows": sorted(
                    [
                        (wf, int(v["count"]))
                        for wf, v in split["workflow_success"].items()
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                ),
                "total_events": int(
                    sum(int(v["count"]) for v in split["workflow_success"].values())
                ),
                "endpoint_usage": dict(sorted(split["endpoint_usage"].items())),
                "difficulty": diff_out,
                "case_studies": split["case_studies"],
            }
        return out


def patch_sentence_encoder_outputs() -> None:
    """Torch 2.9 compatibility: ensure embeddings are regular detached tensors."""
    if getattr(mar_llm_embedding.SentenceEncoder, "_wae_patched", False):
        return
    orig_forward = mar_llm_embedding.SentenceEncoder.forward

    def _patched_forward(self, sentence):
        out = orig_forward(self, sentence)
        if isinstance(out, torch.Tensor):
            return out.clone().detach()
        return out

    mar_llm_embedding.SentenceEncoder.forward = _patched_forward
    mar_llm_embedding.SentenceEncoder._wae_patched = True
