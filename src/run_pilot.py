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


def train_or_eval_epoch(
    *,
    mode: str,
    router,
    data: List[Sample],
    tasks: List[Dict[str, str]],
    reasonings: List[Dict[str, str]],
    optimizer,
    monitor: PilotMonitor,
    prompt_file: str,
    batch_size: int,
    cost_rate: float,
    exec_timeout_s: int,
    batch_timeout_s: int = 180,
    force_workflow_id: Optional[str] = None,
    split_name: str = "",
    difficulty_thr: int = 0,
    routing_tracker: Optional[RoutingTracker] = None,
    workflow_call_budget: Optional[Dict[str, int]] = None,
    cheap_first_cheap_workflow_id: Optional[str] = None,
    cheap_first_premium_workflow_id: Optional[str] = None,
    hardcase_predictor: Optional[Dict[str, object]] = None,
    hardcase_tau: float = 0.5,
    hardcase_cheap_workflow_id: Optional[str] = None,
    hardcase_premium_workflow_id: Optional[str] = None,
    sample_trace_path: Optional[str] = None,
    run_id: str = "",
    workflow_entropy_reg: float = 0.0,
    train: bool = True,
) -> Dict[str, float]:
    total_solved = 0
    total_count = 0
    latencies = []
    router_overheads = []
    llm_infer_latencies = []
    test_exec_latencies = []
    costs_all = []
    utilities = []
    all_loss = []
    call_count_est = 0

    if train:
        router.train()
    else:
        router.eval()

    batches = batched(data, batch_size)
    phase_name = phase_name_from_split(train=train, split_name=split_name)
    os.environ["WAE_PREMIUM_PHASE"] = phase_name
    os.environ["WAE_PREMIUM_SPLIT"] = split_name
    for bi, batch in enumerate(batches):
        queries = [x.query for x in batch]
        task_labels = [2 for _ in batch]  # Code domain
        tasks_y = torch.tensor(task_labels).to(router.device)
        step_start = time.time()
        before_cost = Cost.instance().value
        forward_elapsed = 0.0
        test_exec_elapsed = 0.0

        selected_workflow_ids = None
        selected_role_names = None
        sample_debug = None
        batch_timed_out = False
        batch_failed = False

        try:
            if mode == "masrouter":
                llms = router._runtime_llm_profile
                with batch_timeout_guard(int(batch_timeout_s)):
                    _t_fw = time.time()
                    out = router.forward(
                        queries,
                        tasks,
                        llms,
                        reasonings,
                        given_task=task_labels,
                        prompt_file=prompt_file,
                    )
                    forward_elapsed += time.time() - _t_fw
                results, costs, log_probs, tasks_probs, vae_loss, agent_num_float = out
                call_count_est += int(agent_num_float.sum().item()) + len(batch)
            else:
                if mode == "wae_cheap_first_escalate":
                    if not cheap_first_cheap_workflow_id or not cheap_first_premium_workflow_id:
                        raise RuntimeError(
                            "cheap_first_escalate mode requires cheap/premium workflow ids"
                        )
                    with batch_timeout_guard(int(batch_timeout_s)):
                        _t_fw = time.time()
                        out1 = router.forward(
                            queries,
                            tasks,
                            reasonings,
                            given_task=task_labels,
                            prompt_file=prompt_file,
                            force_workflow_id=cheap_first_cheap_workflow_id,
                        )
                        forward_elapsed += time.time() - _t_fw
                    (
                        results_1,
                        costs_1,
                        log_probs,
                        tasks_probs,
                        vae_loss,
                        agent_num_float,
                        selected_workflow_ids_1,
                        selected_role_names_1,
                        sample_debug_1,
                    ) = out1
                    _t_test = time.time()
                    solved_stage1 = evaluate_batch_outputs(
                        results_1, batch, exec_timeout_s=exec_timeout_s
                    )
                    test_exec_elapsed += time.time() - _t_test

                    results = list(results_1)
                    costs = [float(x) for x in costs_1]
                    solved_list = list(solved_stage1)
                    selected_workflow_ids = [list(x) for x in selected_workflow_ids_1]
                    selected_role_names = [list(x) for x in selected_role_names_1]
                    sample_debug = [dict(x) for x in sample_debug_1]

                    unsolved_idx = [i for i, s in enumerate(solved_stage1) if int(s) == 0]
                    if unsolved_idx:
                        sub_queries = [queries[i] for i in unsolved_idx]
                        sub_task_labels = [task_labels[i] for i in unsolved_idx]
                        sub_batch = [batch[i] for i in unsolved_idx]
                        with batch_timeout_guard(int(batch_timeout_s)):
                            _t_fw = time.time()
                            out2 = router.forward(
                                sub_queries,
                                tasks,
                                reasonings,
                                given_task=sub_task_labels,
                                prompt_file=prompt_file,
                                force_workflow_id=cheap_first_premium_workflow_id,
                            )
                            forward_elapsed += time.time() - _t_fw
                        (
                            results_2,
                            costs_2,
                            _log_probs_2,
                            _tasks_probs_2,
                            _vae_loss_2,
                            _agent_num_float_2,
                            selected_workflow_ids_2,
                            selected_role_names_2,
                            sample_debug_2,
                        ) = out2
                        _t_test = time.time()
                        solved_stage2 = evaluate_batch_outputs(
                            results_2, sub_batch, exec_timeout_s=exec_timeout_s
                        )
                        test_exec_elapsed += time.time() - _t_test
                        for local_i, batch_i in enumerate(unsolved_idx):
                            results[batch_i] = results_2[local_i]
                            solved_list[batch_i] = int(solved_stage2[local_i])
                            costs[batch_i] = float(costs[batch_i]) + float(costs_2[local_i])
                            selected_workflow_ids[batch_i] = (
                                list(selected_workflow_ids[batch_i])
                                + list(selected_workflow_ids_2[local_i])
                            )
                            selected_role_names[batch_i] = (
                                list(selected_role_names[batch_i])
                                + list(selected_role_names_2[local_i])
                            )
                            stage1 = dict(sample_debug[batch_i])
                            stage2 = dict(sample_debug_2[local_i])
                            sample_debug[batch_i] = {
                                "query_hash": stage1.get("query_hash", ""),
                                "escalated": True,
                                "stage1": stage1,
                                "stage2": stage2,
                                "selected_workflow_ids": list(selected_workflow_ids[batch_i]),
                                "selected_roles": list(selected_role_names[batch_i]),
                                "invoked_workflow_ids": list(
                                    dict.fromkeys(
                                        list(stage1.get("invoked_workflow_ids", []))
                                        + list(stage2.get("invoked_workflow_ids", []))
                                    )
                                ),
                                "selected_base_models": list(
                                    dict.fromkeys(
                                        list(stage1.get("selected_base_models", []))
                                        + list(stage2.get("selected_base_models", []))
                                    )
                                ),
                                "selected_endpoint_names": list(
                                    dict.fromkeys(
                                        list(stage1.get("selected_endpoint_names", []))
                                        + list(stage2.get("selected_endpoint_names", []))
                                    )
                                ),
                                "selected_model_ids": list(
                                    dict.fromkeys(
                                        list(stage1.get("selected_model_ids", []))
                                        + list(stage2.get("selected_model_ids", []))
                                    )
                                ),
                                "prompt_hashes": list(stage1.get("prompt_hashes", []))
                                + list(stage2.get("prompt_hashes", [])),
                                "prompt_template_hash": str(
                                    stage1.get("prompt_template_hash", "")
                                    or stage2.get("prompt_template_hash", "")
                                ),
                                "output_hash": str(
                                    stage2.get("output_hash", "")
                                    or stage1.get("output_hash", "")
                                ),
                                "prompt_tokens": int(stage1.get("prompt_tokens", 0))
                                + int(stage2.get("prompt_tokens", 0)),
                                "completion_tokens": int(stage1.get("completion_tokens", 0))
                                + int(stage2.get("completion_tokens", 0)),
                                "llm_infer_latency_s": float(
                                    stage1.get("llm_infer_latency_s", 0.0)
                                )
                                + float(stage2.get("llm_infer_latency_s", 0.0)),
                                "overflow_retries": int(stage1.get("overflow_retries", 0))
                                + int(stage2.get("overflow_retries", 0)),
                                "endpoint_calls": int(stage1.get("endpoint_calls", 0))
                                + int(stage2.get("endpoint_calls", 0)),
                            }
                    for i in range(len(sample_debug)):
                        if "selected_workflow_ids" not in sample_debug[i]:
                            sample_debug[i]["query_hash"] = sample_debug[i].get("query_hash", "")
                            sample_debug[i]["escalated"] = False
                            sample_debug[i]["selected_workflow_ids"] = list(
                                selected_workflow_ids[i]
                            )
                            sample_debug[i]["selected_roles"] = list(selected_role_names[i])
                            sample_debug[i]["invoked_workflow_ids"] = list(
                                selected_workflow_ids[i]
                            )
                            sample_debug[i]["prompt_template_hash"] = str(
                                sample_debug[i].get("prompt_template_hash", "")
                            )
                            sample_debug[i]["llm_infer_latency_s"] = float(
                                sample_debug[i].get("llm_infer_latency_s", 0.0)
                            )
                    costs = [float(x) for x in costs]
                elif mode == "wae_dynamic_hardcase_gate":
                    if not hardcase_cheap_workflow_id or not hardcase_premium_workflow_id:
                        raise RuntimeError(
                            "wae_dynamic_hardcase_gate requires cheap/premium workflow ids"
                        )
                    results = ["" for _ in batch]
                    costs = [0.0 for _ in batch]
                    solved_list = [0 for _ in batch]
                    selected_workflow_ids = [[] for _ in batch]
                    selected_role_names = [[] for _ in batch]
                    sample_debug = [{} for _ in batch]

                    premium_idx = []
                    cheap_idx = []
                    fail_probs: List[float] = []
                    for i, sample in enumerate(batch):
                        p_fail = hardcase_fail_prob(sample, hardcase_predictor)
                        fail_probs.append(float(p_fail))
                        if p_fail >= float(hardcase_tau):
                            premium_idx.append(i)
                        else:
                            cheap_idx.append(i)

                    def _run_subset(
                        subset_idx: List[int], forced_wf: str
                    ) -> None:
                        nonlocal forward_elapsed, test_exec_elapsed
                        if not subset_idx:
                            return
                        sub_queries = [queries[i] for i in subset_idx]
                        sub_task_labels = [task_labels[i] for i in subset_idx]
                        sub_batch = [batch[i] for i in subset_idx]
                        with batch_timeout_guard(int(batch_timeout_s)):
                            _t_fw = time.time()
                            out_sub = router.forward(
                                sub_queries,
                                tasks,
                                reasonings,
                                given_task=sub_task_labels,
                                prompt_file=prompt_file,
                                force_workflow_id=forced_wf,
                            )
                            forward_elapsed += time.time() - _t_fw
                        (
                            sub_results,
                            sub_costs,
                            _sub_log_probs,
                            _sub_tasks_probs,
                            _sub_vae_loss,
                            _sub_agent_num_float,
                            sub_selected_workflow_ids,
                            sub_selected_role_names,
                            sub_sample_debug,
                        ) = out_sub
                        _t_test = time.time()
                        sub_solved = evaluate_batch_outputs(
                            sub_results, sub_batch, exec_timeout_s=exec_timeout_s
                        )
                        test_exec_elapsed += time.time() - _t_test
                        for local_i, batch_i in enumerate(subset_idx):
                            results[batch_i] = sub_results[local_i]
                            costs[batch_i] = float(sub_costs[local_i])
                            solved_list[batch_i] = int(sub_solved[local_i])
                            selected_workflow_ids[batch_i] = list(
                                sub_selected_workflow_ids[local_i]
                            )
                            selected_role_names[batch_i] = list(
                                sub_selected_role_names[local_i]
                            )
                            dbg = dict(sub_sample_debug[local_i])
                            dbg["hardcase_fail_prob"] = float(fail_probs[batch_i])
                            dbg["hardcase_gate_choice"] = (
                                "premium"
                                if batch_i in premium_idx
                                else "cheap"
                            )
                            sample_debug[batch_i] = dbg

                    _run_subset(cheap_idx, hardcase_cheap_workflow_id)
                    _run_subset(premium_idx, hardcase_premium_workflow_id)
                    log_probs = [torch.tensor(0.0, device=router.device) for _ in batch]
                    tasks_probs = torch.zeros((len(batch), len(tasks)), device=router.device)
                    vae_loss = torch.zeros((len(batch),), device=router.device)
                    agent_num_float = torch.ones((len(batch),), device=router.device)
                    costs = [float(x) for x in costs]

                    wf_budget = workflow_call_budget or {}
                    for wf_names in selected_workflow_ids:
                        for wf_name in wf_names:
                            wf_id = wf_name.split("wf::", 1)[-1]
                            call_count_est += wf_budget.get(wf_id, 1)
                        call_count_est += 1
                else:
                    with batch_timeout_guard(int(batch_timeout_s)):
                        _t_fw = time.time()
                        out = router.forward(
                            queries,
                            tasks,
                            reasonings,
                            given_task=task_labels,
                            prompt_file=prompt_file,
                            force_workflow_id=force_workflow_id,
                        )
                        forward_elapsed += time.time() - _t_fw
                    (
                        results,
                        costs,
                        log_probs,
                        tasks_probs,
                        vae_loss,
                        agent_num_float,
                        selected_workflow_ids,
                        selected_role_names,
                        sample_debug,
                    ) = out
                    _t_test = time.time()
                    solved_list = evaluate_batch_outputs(
                        results, batch, exec_timeout_s=exec_timeout_s
                    )
                    test_exec_elapsed += time.time() - _t_test
                wf_budget = workflow_call_budget or {}
                for wf_names in selected_workflow_ids:
                    for wf_name in wf_names:
                        wf_id = wf_name.split("wf::", 1)[-1]
                        call_count_est += wf_budget.get(wf_id, 1)
                    call_count_est += 1  # final decision node
        except BatchTimeoutError:
            batch_timed_out = True
            logger.error(
                "Batch timeout at split={} batch={}/{} mode={} timeout={}s; fallback applied.",
                split_name,
                bi + 1,
                len(batches),
                mode,
                int(batch_timeout_s),
            )
            results = ["" for _ in batch]
            costs = [0.0 for _ in batch]
            log_probs = [torch.tensor(0.0, device=router.device) for _ in batch]
            tasks_probs = torch.zeros((len(batch), len(tasks)), device=router.device)
            vae_loss = torch.zeros((len(batch),), device=router.device)
            agent_num_float = torch.ones((len(batch),), device=router.device)
            solved_list = [0 for _ in batch]
            if mode != "masrouter":
                selected_workflow_ids = [["wf::timeout"] for _ in batch]
                selected_role_names = [["TimeoutRole"] for _ in batch]
                sample_debug = [
                    {
                        "query_hash": hashlib.md5(s.query.encode("utf-8")).hexdigest()[:10],
                        "invoked_workflow_ids": ["wf::timeout"],
                        "selected_base_models": [],
                        "selected_endpoint_names": [],
                        "selected_model_ids": [],
                        "prompt_hashes": [],
                        "prompt_template_hash": "",
                        "output_hash": "",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "llm_infer_latency_s": 0.0,
                        "overflow_retries": 0,
                        "endpoint_calls": 0,
                    }
                    for s in batch
                ]
            call_count_est += len(batch)
            monitor.heartbeat(
                note=(
                    f"batch-timeout split={split_name} batch={bi+1}/{len(batches)} "
                    f"mode={mode} timeout_s={int(batch_timeout_s)}"
                )
            )
        except Exception as e:
            batch_failed = True
            err_text = f"{type(e).__name__}: {str(e)}"
            logger.error(
                "Batch error at split={} batch={}/{} mode={}: {}",
                split_name,
                bi + 1,
                len(batches),
                mode,
                err_text,
            )
            logger.debug(traceback.format_exc())
            results = ["" for _ in batch]
            costs = [0.0 for _ in batch]
            log_probs = [torch.tensor(0.0, device=router.device) for _ in batch]
            tasks_probs = torch.zeros((len(batch), len(tasks)), device=router.device)
            vae_loss = torch.zeros((len(batch),), device=router.device)
            agent_num_float = torch.ones((len(batch),), device=router.device)
            solved_list = [0 for _ in batch]
            if mode != "masrouter":
                selected_workflow_ids = [["wf::error"] for _ in batch]
                selected_role_names = [["ErrorRole"] for _ in batch]
                sample_debug = [
                    {
                        "query_hash": hashlib.md5(s.query.encode("utf-8")).hexdigest()[:10],
                        "invoked_workflow_ids": ["wf::error"],
                        "selected_base_models": [],
                        "selected_endpoint_names": [],
                        "selected_model_ids": [],
                        "prompt_hashes": [],
                        "prompt_template_hash": "",
                        "output_hash": "",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "llm_infer_latency_s": 0.0,
                        "overflow_retries": 0,
                        "endpoint_calls": 0,
                    }
                    for s in batch
                ]
            call_count_est += len(batch)
            monitor.record_error(
                f"batch-error split={split_name} batch={bi+1}/{len(batches)} mode={mode} {err_text}"
            )
            monitor.heartbeat(
                note=(
                    f"batch-error split={split_name} batch={bi+1}/{len(batches)} "
                    f"mode={mode} error={type(e).__name__}"
                )
            )

        if mode == "masrouter":
            if not batch_timed_out and not batch_failed:
                _t_test = time.time()
                solved_list = evaluate_batch_outputs(
                    results, batch, exec_timeout_s=exec_timeout_s
                )
                test_exec_elapsed += time.time() - _t_test
        total_solved += sum(solved_list)
        total_count += len(solved_list)
        step_latency = time.time() - step_start

        if routing_tracker is not None and selected_workflow_ids is not None and selected_role_names is not None:
            sample_latency = step_latency / max(len(batch), 1)
            sample_test_exec = test_exec_elapsed / max(len(batch), 1)
            for sample_idx, (sample, solved, sample_cost, wf_names, role_names) in enumerate(
                zip(
                    batch,
                    solved_list,
                    costs,
                    selected_workflow_ids,
                    selected_role_names,
                )
            ):
                dbg = sample_debug[sample_idx] if sample_debug else {}
                sample_llm_infer = float(dbg.get("llm_infer_latency_s", 0.0))
                sample_router_overhead = max(
                    float(sample_latency) - float(sample_test_exec) - float(sample_llm_infer),
                    0.0,
                )
                llm_infer_latencies.append(sample_llm_infer)
                test_exec_latencies.append(float(sample_test_exec))
                router_overheads.append(float(sample_router_overhead))
                routing_tracker.record(
                    split_name=split_name,
                    sample=sample,
                    selected_workflows=list(wf_names),
                    selected_roles=list(role_names),
                    solved=int(solved),
                    sample_cost=float(sample_cost),
                    sample_latency_s=float(sample_latency),
                    threshold=int(difficulty_thr),
                )
                if sample_trace_path:
                    output_hash = short_hash(str(results[sample_idx]))
                    rec = {
                        "ts_utc": datetime.utcnow().isoformat(),
                        "run_id": run_id,
                        "mode": mode,
                        "phase": phase_name,
                        "split": split_name,
                        "task": dbg.get("task", ""),
                        "collab": dbg.get("collab", ""),
                        "source": sample.source,
                        "query_hash": dbg.get("query_hash", hashlib.md5(sample.query.encode("utf-8")).hexdigest()[:10]),
                        "chosen_workflow_id": wf_names[0] if len(wf_names) == 1 else wf_names,
                        "chosen_workflow_ids": list(wf_names),
                        "invoked_workflow_ids": list(
                            dbg.get("invoked_workflow_ids", list(wf_names))
                        ),
                        "selected_roles": list(role_names),
                        "base_models": dbg.get("selected_base_models", []),
                        "base_endpoint_name": dbg.get("selected_endpoint_names", []),
                        "model_id": dbg.get("selected_model_ids", []),
                        "prompt_template_hash": dbg.get("prompt_template_hash", ""),
                        "prompt_hashes": dbg.get("prompt_hashes", []),
                        "output_hash": output_hash,
                        "output_hash_from_router": dbg.get("output_hash", ""),
                        "prompt_tokens": int(dbg.get("prompt_tokens", 0)),
                        "completion_tokens": int(dbg.get("completion_tokens", 0)),
                        "llm_infer_latency_s": float(sample_llm_infer),
                        "test_exec_latency_s": float(sample_test_exec),
                        "router_overhead_latency_s": float(sample_router_overhead),
                        "max_tokens_auto_reduce_retries": int(dbg.get("overflow_retries", 0)),
                        "endpoint_calls": int(dbg.get("endpoint_calls", 0)),
                        "hardcase_fail_prob": dbg.get("hardcase_fail_prob", None),
                        "hardcase_gate_choice": dbg.get("hardcase_gate_choice", None),
                        "escalated": bool(dbg.get("escalated", False)),
                        "pass": int(solved),
                        "sample_cost": float(sample_cost),
                        "sample_latency_s": float(sample_latency),
                    }
                    with open(sample_trace_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        elif mode == "masrouter":
            sample_latency = step_latency / max(len(batch), 1)
            sample_test_exec = test_exec_elapsed / max(len(batch), 1)
            sample_llm_infer = forward_elapsed / max(len(batch), 1)
            sample_router_overhead = max(
                float(sample_latency) - float(sample_test_exec) - float(sample_llm_infer),
                0.0,
            )
            for _ in batch:
                llm_infer_latencies.append(float(sample_llm_infer))
                test_exec_latencies.append(float(sample_test_exec))
                router_overheads.append(float(sample_router_overhead))

        latencies.append(step_latency / max(len(batch), 1))
        costs_all.extend(list(costs))

        sample_util = []
        answer_losses = []
        for solved, log_prob, cost in zip(solved_list, log_probs, costs):
            utility = float(solved) - float(cost) * float(cost_rate)
            sample_util.append(utility)
            answer_losses.append(-log_prob * utility)
            utilities.append(utility)
            monitor.record_step(
                latency_s=step_latency / max(len(batch), 1),
                cost=float(cost),
                utility=float(utility),
            )

        if (
            train
            and mode != "wae_cheap_first_escalate"
            and not batch_timed_out
            and not batch_failed
        ):
            optimizer.zero_grad()
            task_loss = F.cross_entropy(tasks_probs, tasks_y)
            answer_loss = torch.stack(answer_losses).sum() / max(len(answer_losses), 1)
            routing_entropy = getattr(router, "last_workflow_entropy", None)
            if routing_entropy is None:
                entropy_bonus = torch.tensor(0.0, device=router.device)
            elif isinstance(routing_entropy, torch.Tensor):
                entropy_bonus = routing_entropy.mean()
            else:
                entropy_bonus = torch.tensor(float(routing_entropy), device=router.device)
            entropy_weight = (
                float(workflow_entropy_reg) if mode.startswith("wae_dynamic") else 0.0
            )
            loss = (
                task_loss
                + answer_loss
                + vae_loss.mean() * 0.001
                - entropy_weight * entropy_bonus
            )
            loss.backward()
            optimizer.step()
            all_loss.append(float(loss.detach().cpu().item()))

        if (bi + 1) % 2 == 0:
            monitor.heartbeat(note=f"batch={bi+1}/{len(batches)} train={train}")

    acc = total_solved / max(total_count, 1)
    return {
        "accuracy_or_pass1": acc,
        "solved": total_solved,
        "count": total_count,
        "avg_cost": sum(costs_all) / max(len(costs_all), 1),
        "total_cost": sum(costs_all),
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "router_overhead_p50_s": percentile(router_overheads, 50),
        "router_overhead_p95_s": percentile(router_overheads, 95),
        "llm_infer_p50_s": percentile(llm_infer_latencies, 50),
        "llm_infer_p95_s": percentile(llm_infer_latencies, 95),
        "test_exec_p50_s": percentile(test_exec_latencies, 50),
        "test_exec_p95_s": percentile(test_exec_latencies, 95),
        "avg_utility": sum(utilities) / max(len(utilities), 1),
        "avg_loss": sum(all_loss) / max(len(all_loss), 1) if all_loss else 0.0,
        "call_count_est": int(call_count_est),
        "prompt_tokens_total": PromptTokens.instance().value,
        "completion_tokens_total": CompletionTokens.instance().value,
    }


def choose_static_workflow(
    metrics: List[Dict[str, float]], mode: str
) -> Optional[str]:
    if not metrics:
        return None
    if mode == "wae_static_cheap":
        return min(metrics, key=lambda x: x["avg_cost"])["id"]
    if mode == "wae_static_premium":
        return max(metrics, key=lambda x: x["pass_rate"])["id"]
    return None


def build_candidates_from_pareto(
    workflow_candidates: List[Dict[str, object]], pareto_ids: List[str]
) -> List[Dict[str, object]]:
    idset = set(pareto_ids)
    cands = [wf for wf in workflow_candidates if wf["id"] in idset]
    return cands if cands else workflow_candidates


def parse_excluded_tiers(raw: str) -> List[str]:
    if not raw:
        return []
    return sorted({x.strip().lower() for x in raw.split(",") if x.strip()})


def filter_workflows_by_tier(
    workflow_candidates: List[Dict[str, object]], excluded_tiers: List[str]
) -> List[Dict[str, object]]:
    if not excluded_tiers:
        return workflow_candidates
    excluded = set(excluded_tiers)
    kept = [
        wf
        for wf in workflow_candidates
        if str(wf.get("budget_tier", "")).lower() not in excluded
    ]
    return kept


def workflow_prior_map(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    priors: Dict[str, float] = {}
    for m in metrics:
        wf_id = str(m["id"])
        p = float(m.get("pass_rate", 0.5))
        priors[wf_id] = max(1e-4, min(1.0 - 1e-4, p))
    return priors


def apply_premium_prior_gating(
    workflow_candidates: List[Dict[str, object]],
    metrics: List[Dict[str, float]],
    cheap_workflow_id: str,
    epsilon: float = 0.02,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    by_id = {str(m["id"]): m for m in metrics}
    cheap_ref = by_id.get(cheap_workflow_id)
    if cheap_ref is None:
        cheap_ref = min(metrics, key=lambda x: float(x.get("avg_cost", 1e9)))
        cheap_workflow_id = str(cheap_ref["id"])

    cheap_prior = float(cheap_ref.get("pass_rate", 0.0))
    cheap_cost = max(float(cheap_ref.get("avg_cost", 0.0)), 1e-9)
    cheap_roi = cheap_prior / cheap_cost

    kept: List[Dict[str, object]] = []
    dropped: List[Dict[str, object]] = []
    for wf in workflow_candidates:
        wf_id = str(wf["id"])
        rec = by_id.get(wf_id, {})
        prior = float(rec.get("pass_rate", 0.5))
        avg_cost = max(float(rec.get("avg_cost", 0.0)), 1e-9)
        roi = prior / avg_cost
        tier = str(wf.get("budget_tier", "")).lower()
        keep = True
        reason = ""
        if tier == "premium":
            if prior < cheap_prior - float(epsilon):
                keep = False
                reason = "prior_below_threshold"
            elif roi < cheap_roi:
                keep = False
                reason = "roi_below_cheap"
        if keep:
            kept.append(wf)
        else:
            dropped.append(
                {
                    "id": wf_id,
                    "tier": tier,
                    "prior": prior,
                    "avg_cost": avg_cost,
                    "roi": roi,
                    "reason": reason,
                }
            )

    if not kept:
        kept = workflow_candidates

    report = {
        "cheap_reference_id": cheap_workflow_id,
        "cheap_reference_prior": cheap_prior,
        "cheap_reference_roi": cheap_roi,
        "gate_type": "prior_and_roi",
        "epsilon": float(epsilon),
        "kept_ids": [str(wf["id"]) for wf in kept],
        "dropped": dropped,
    }
    return kept, report


def apply_premium_roi_gating(
    workflow_candidates: List[Dict[str, object]],
    metrics: List[Dict[str, float]],
    cheap_workflow_id: str,
    roi_margin: float = 0.0,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    by_id = {str(m["id"]): m for m in metrics}
    cheap_ref = by_id.get(cheap_workflow_id)
    if cheap_ref is None:
        cheap_ref = min(metrics, key=lambda x: float(x.get("avg_cost", 1e9)))
        cheap_workflow_id = str(cheap_ref["id"])

    cheap_prior = float(cheap_ref.get("pass_rate", 0.0))
    cheap_cost = max(float(cheap_ref.get("avg_cost", 0.0)), 1e-9)
    cheap_roi = cheap_prior / cheap_cost
    roi_threshold = cheap_roi * (1.0 + float(roi_margin))

    kept: List[Dict[str, object]] = []
    dropped: List[Dict[str, object]] = []
    for wf in workflow_candidates:
        wf_id = str(wf["id"])
        rec = by_id.get(wf_id, {})
        prior = float(rec.get("pass_rate", 0.5))
        avg_cost = max(float(rec.get("avg_cost", 0.0)), 1e-9)
        roi = prior / avg_cost
        tier = str(wf.get("budget_tier", "")).lower()
        keep = True
        reason = ""
        if tier == "premium" and roi < roi_threshold:
            keep = False
            reason = "roi_below_threshold"
        if keep:
            kept.append(wf)
        else:
            dropped.append(
                {
                    "id": wf_id,
                    "tier": tier,
                    "prior": prior,
                    "avg_cost": avg_cost,
                    "roi": roi,
                    "reason": reason,
                }
            )
    if not kept:
        kept = workflow_candidates

    report = {
        "gate_type": "roi_only",
        "cheap_reference_id": cheap_workflow_id,
        "cheap_reference_prior": cheap_prior,
        "cheap_reference_roi": cheap_roi,
        "roi_margin": float(roi_margin),
        "roi_threshold": float(roi_threshold),
        "kept_ids": [str(wf["id"]) for wf in kept],
        "dropped": dropped,
    }
    return kept, report


def workflow_call_budget_map(
    workflow_candidates: List[Dict[str, object]]
) -> Dict[str, int]:
    method_budget = {
        "io": 1,
        "refine2": 2,
        "self_consistency3": 3,
        "critique_refine": 3,
        "gen_test_select3": 3,
    }
    out: Dict[str, int] = {}
    for wf in workflow_candidates:
        wf_id = str(wf["id"])
        method = str(wf.get("method", "io"))
        out[wf_id] = int(method_budget.get(method, 1))
    return out


def collect_workflow_fail_labels(
    *,
    router,
    data: List[Sample],
    tasks: List[Dict[str, str]],
    reasonings: List[Dict[str, str]],
    prompt_file: str,
    workflow_id: str,
    batch_size: int,
    exec_timeout_s: int,
    batch_timeout_s: int,
) -> List[int]:
    if not data:
        return []
    labels: List[int] = []
    batches = batched(data, max(int(batch_size), 1))
    for batch in batches:
        queries = [x.query for x in batch]
        task_labels = [2 for _ in batch]
        with torch.no_grad():
            with batch_timeout_guard(int(batch_timeout_s)):
                out = router.forward(
                    queries,
                    tasks,
                    reasonings,
                    given_task=task_labels,
                    prompt_file=prompt_file,
                    force_workflow_id=workflow_id,
                )
        results = out[0]
        solved = evaluate_batch_outputs(results, batch, exec_timeout_s=exec_timeout_s)
        labels.extend([0 if int(v) == 1 else 1 for v in solved])  # fail=1
    return labels


def main() -> None:
    args, parser = parse_args()
    exp_cfg = load_exp_config(args.experiment_config)
    for k, v in exp_cfg.items():
        if not hasattr(args, k):
            continue
        if getattr(args, k) == parser.get_default(k):
            setattr(args, k, v)

    fix_random_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.deterministic_inference:
        os.environ["WAE_DEFAULT_TEMPERATURE"] = "0.0"
        os.environ["WAE_DEFAULT_TOP_P"] = "1.0"
        os.environ["WAE_REQUEST_SEED"] = str(args.seed)
        os.environ["WAE_ROUTER_GREEDY"] = "1"
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        os.environ.pop("WAE_REQUEST_SEED", None)
        os.environ.pop("WAE_ROUTER_GREEDY", None)
    if args.deterministic_router_components:
        os.environ["WAE_ROUTER_GREEDY_ALL"] = "1"
    else:
        os.environ.pop("WAE_ROUTER_GREEDY_ALL", None)

    run_id = args.run_id or f"{args.mode}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    dirs = setup_run_dirs(run_id)
    monitor = PilotMonitor(run_dir=dirs["root"])
    monitor.set_stage("setup")
    if args.mode != "masrouter":
        os.environ["WAE_PREMIUM_DEBUG_LOG"] = os.path.join(
            dirs["logs"], "premium_debug.jsonl"
        )
    else:
        os.environ.pop("WAE_PREMIUM_DEBUG_LOG", None)

    endpoint_manager = EndpointManager(args.model_endpoints)
    os.environ["WAE_PREMIUM_REQUIRE_TESTS"] = "1" if int(args.premium_require_tests) else "0"
    workflow_candidates = default_workflow_candidates()
    if args.mode == "wae_dynamic_no_premium":
        existing = parse_excluded_tiers(args.exclude_budget_tiers)
        if "premium" not in existing:
            existing.append("premium")
        args.exclude_budget_tiers = ",".join(sorted(set(existing)))
    if args.mode == "wae_dynamic_hardcase_gate" and args.epochs > 0:
        logger.warning("wae_dynamic_hardcase_gate is heuristic routing. Setting epochs=0.")
        args.epochs = 0
    if args.mode == "wae_cheap_first_escalate" and args.epochs > 0:
        logger.warning("cheap_first_escalate is heuristic-only. Setting epochs=0.")
        args.epochs = 0

    excluded_tiers = parse_excluded_tiers(args.exclude_budget_tiers)
    if excluded_tiers:
        workflow_candidates = filter_workflows_by_tier(
            workflow_candidates, excluded_tiers
        )
        if not workflow_candidates:
            raise RuntimeError(
                f"All workflows filtered out by exclude tiers: {excluded_tiers}"
            )

    # Annotate workflow candidates with endpoint model ids for traceability.
    specs = endpoint_manager.endpoint_specs
    for wf in workflow_candidates:
        base_name = str(wf.get("base_model", ""))
        wf["base_model_id"] = str(specs.get(base_name, {}).get("model_id", "unknown"))

    if args.mode == "masrouter":
        required_endpoints = endpoint_manager.endpoint_names
    else:
        required_endpoints = sorted(set(str(wf["base_model"]) for wf in workflow_candidates))
        missing_workflow_mappings = endpoint_manager.verify_workflow_mappings(workflow_candidates)
        if missing_workflow_mappings:
            raise RuntimeError(
                f"Workflow->endpoint mapping missing for workflows: {missing_workflow_mappings}"
            )

    monitor.set_stage("serving_gate")
    endpoint_gate = endpoint_manager.check_ready(
        required_names=required_endpoints,
        retries=args.endpoint_ready_retries,
        interval_s=args.endpoint_ready_interval_s,
        warmup=args.endpoint_warmup,
    )
    save_json(os.path.join(dirs["metrics"], "endpoint_gate.json"), endpoint_gate)
    if args.no_fallback and not endpoint_gate["all_ready"]:
        raise RuntimeError(f"Serving gate failed with --no_fallback: {endpoint_gate}")
    if not endpoint_gate["all_ready"]:
        logger.warning("Serving gate not fully ready. Continuing because --no_fallback is not set.")

    hetero_summary = endpoint_manager.heterogeneity_summary(required_endpoints)
    save_json(os.path.join(dirs["metrics"], "endpoint_heterogeneity.json"), hetero_summary)
    if args.require_heterogeneous_endpoints:
        # Require diversity across all required endpoints. When only 2 endpoints
        # are required by the active workflow set, demanding 3 unique backends
        # is impossible and should not fail-fast.
        min_unique = min(3, len(required_endpoints))
        if (
            hetero_summary["num_unique_model_ids"] < min_unique
            or hetero_summary["num_unique_base_urls"] < min_unique
        ):
            raise RuntimeError(
                "Heterogeneous endpoint requirement failed. "
                f"required_unique={min_unique}, summary={hetero_summary}"
            )

    runtime_patch = LLMRuntimePatch(endpoint_manager)
    runtime_patch.install()
    patch_sentence_encoder_outputs()
    # MasRouter code uses project-relative paths such as "MAR/Roles", so
    # the runner cwd must be the MasRouter checkout root.
    os.chdir(os.environ.get("MASROUTER_PATH", "/workspace/masrouter"))

    train_mbpp, test_mbpp = load_mbpp_samples(args.train_samples, args.test_samples_mbpp)
    if args.inject_tests_into_humaneval_query:
        test_humaneval = load_humaneval_samples_with_inline_tests(
            args.test_samples_humaneval
        )
    else:
        test_humaneval = load_humaneval_samples(args.test_samples_humaneval)
    tasks = TASKS_PROFILE
    reasonings = REASONING_PROFILE
    _mar_root = os.environ.get("MASROUTER_PATH", "/workspace/masrouter")
    prompt_file = f"{_mar_root}/MAR/Roles/FinalNode/mbpp.json"
    prompt_file_humaneval = f"{_mar_root}/MAR/Roles/FinalNode/humaneval.json"

    monitor.set_stage("router_init")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = None
    force_workflow_id = args.force_workflow_id
    if args.mode == "wae_static_cheap" and force_workflow_id is None:
        force_workflow_id = args.cheap_workflow_id
    if args.mode == "wae_static_premium" and force_workflow_id is None:
        force_workflow_id = args.premium_workflow_id

    # Gate: offline Pareto only for WaE modes. For forced static workflow runs,
    # skip expensive calibration and directly pin the requested workflow.
    pareto_metrics = []
    pareto_json = None
    prior_report = None
    workflow_priors: Dict[str, float] = {}
    skip_offline_pareto = (
        (
            args.mode in ("wae_static_cheap", "wae_static_premium")
            and force_workflow_id is not None
        )
        or (args.mode == "wae_dynamic_no_premium")
        or (args.mode == "wae_dynamic_hardcase_gate")
        or (args.mode == "wae_cheap_first_escalate")
    )
    if args.mode != "masrouter" and not skip_offline_pareto:
        monitor.set_stage("offline_pareto")
        pareto_json = os.path.join(dirs["metrics"], "role_pareto_library.json")
        cache_path = str(args.reuse_pareto_json or "").strip()
        cache_loaded = False
        if cache_path and os.path.exists(cache_path):
            payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
            pareto_metrics = list(payload.get("workflow_metrics", []))
            pareto_ids = list(payload.get("pareto_front", []))
            save_json(pareto_json, payload)
            cache_loaded = True
            monitor.heartbeat(note=f"offline_pareto cache loaded: {cache_path}")
        else:
            pareto_metrics, role_lib = build_role_conditioned_library(
                endpoint_manager=endpoint_manager,
                workflow_candidates=workflow_candidates,
                calibration_size=args.calibration_size,
                output_path=pareto_json,
                exec_timeout_s=args.exec_timeout_s,
                progress_hook=monitor.heartbeat,
                calibration_mix=args.calibration_mix,
                seed=args.seed,
            )
            payload = json.loads(Path(pareto_json).read_text(encoding="utf-8"))
            pareto_ids = list(payload["pareto_front"])
            if cache_path:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                save_json(cache_path, payload)
                monitor.heartbeat(note=f"offline_pareto cache written: {cache_path}")
        if force_workflow_id:
            # Keep forced workflow available even if excluded by Pareto filtering.
            keep_ids = sorted(set(pareto_ids + [force_workflow_id]))
            workflow_candidates = build_candidates_from_pareto(workflow_candidates, keep_ids)
        else:
            workflow_candidates = build_candidates_from_pareto(workflow_candidates, pareto_ids)
        if args.mode in ("wae_static_cheap", "wae_static_premium") and force_workflow_id is None:
            force_workflow_id = choose_static_workflow(pareto_metrics, args.mode)
        if force_workflow_id:
            valid_ids = {str(wf["id"]) for wf in workflow_candidates}
            if force_workflow_id not in valid_ids:
                logger.warning(
                    "Forced workflow id '{}' is unavailable after filtering. Falling back to auto-select.",
                    force_workflow_id,
                )
                force_workflow_id = choose_static_workflow(pareto_metrics, args.mode)
        if args.mode == "wae_dynamic_prior_gated":
            workflow_priors = workflow_prior_map(pareto_metrics)
            workflow_candidates, prior_report = apply_premium_prior_gating(
                workflow_candidates=workflow_candidates,
                metrics=pareto_metrics,
                cheap_workflow_id=args.cheap_workflow_id,
                epsilon=float(args.premium_prior_epsilon),
            )
            # Keep prior map only for active candidates.
            kept_ids = {str(wf["id"]) for wf in workflow_candidates}
            workflow_priors = {
                wf_id: workflow_priors[wf_id]
                for wf_id in workflow_priors
                if wf_id in kept_ids
            }
        elif args.mode == "wae_dynamic_roi_gated":
            workflow_candidates, prior_report = apply_premium_roi_gating(
                workflow_candidates=workflow_candidates,
                metrics=pareto_metrics,
                cheap_workflow_id=args.cheap_workflow_id,
                roi_margin=float(args.roi_gate_margin),
            )
    elif args.mode != "masrouter" and skip_offline_pareto:
        monitor.set_stage("offline_pareto_skip")
        valid_ids = {str(wf["id"]) for wf in workflow_candidates}
        if args.mode == "wae_cheap_first_escalate":
            keep_ids = {str(args.cheap_workflow_id), str(args.premium_workflow_id)}
            missing = sorted([wf_id for wf_id in keep_ids if wf_id not in valid_ids])
            if missing:
                raise RuntimeError(
                    f"cheap_first_escalate workflow ids are missing: {missing}"
                )
            workflow_candidates = [
                wf for wf in workflow_candidates if str(wf["id"]) in keep_ids
            ]
            monitor.heartbeat("offline_pareto skipped (cheap-first fixed workflows)")
        elif args.mode == "wae_dynamic_no_premium":
            monitor.heartbeat("offline_pareto skipped (dynamic_no_premium uses filtered workflow set)")
        elif args.mode == "wae_dynamic_hardcase_gate":
            keep_ids = {
                str(args.cheap_workflow_id),
                str(args.hardcase_premium_workflow_id),
            }
            missing = sorted([wf_id for wf_id in keep_ids if wf_id not in valid_ids])
            if missing:
                raise RuntimeError(
                    f"hardcase_gate workflow ids are missing: {missing}"
                )
            workflow_candidates = [
                wf for wf in workflow_candidates if str(wf["id"]) in keep_ids
            ]
            monitor.heartbeat(
                "offline_pareto skipped (hardcase gate fixed cheap/premium workflows)"
            )
        else:
            if force_workflow_id not in valid_ids:
                raise RuntimeError(
                    f"Forced workflow '{force_workflow_id}' is not present in workflow candidates."
                )
            workflow_candidates = [wf for wf in workflow_candidates if str(wf["id"]) == force_workflow_id]
            monitor.heartbeat("offline_pareto skipped (forced static workflow)")

    if args.mode == "masrouter":
        router = MasRouter(max_agent=args.max_agent, device=device).to(device)
        router._runtime_llm_profile = endpoint_manager.as_llm_profile()
    else:
        router = WaERouter(
            workflow_candidates=workflow_candidates,
            max_agent=args.max_agent,
            workflow_prior=workflow_priors if args.mode == "wae_dynamic_prior_gated" else None,
            workflow_prior_beta=float(args.workflow_prior_beta)
            if args.mode == "wae_dynamic_prior_gated"
            else 0.0,
            device=device,
        ).to(device)
    routing_tracker = RoutingTracker(workflow_candidates) if args.mode != "masrouter" else None
    wf_call_budget = workflow_call_budget_map(workflow_candidates)
    hardcase_predictor: Optional[Dict[str, object]] = None
    if args.mode == "wae_dynamic_hardcase_gate":
        monitor.set_stage("hardcase_calibration")
        cal_samples = build_hardcase_calibration_set(
            train_mbpp=train_mbpp,
            test_humaneval=test_humaneval,
            calibration_size=int(args.hardcase_calibration_size),
            mix=str(args.calibration_mix),
            seed=int(args.seed),
        )
        fail_labels = collect_workflow_fail_labels(
            router=router,
            data=cal_samples,
            tasks=tasks,
            reasonings=reasonings,
            prompt_file=prompt_file,
            workflow_id=str(args.cheap_workflow_id),
            batch_size=max(1, int(args.batch_size)),
            exec_timeout_s=int(args.exec_timeout_s),
            batch_timeout_s=int(args.batch_timeout_s),
        )
        hardcase_predictor = fit_hardcase_logistic(
            samples=cal_samples,
            cheap_fail_labels=fail_labels,
            seed=int(args.seed),
        )
        hardcase_predictor["tau"] = float(args.hardcase_tau)
        hardcase_predictor["cheap_workflow_id"] = str(args.cheap_workflow_id)
        hardcase_predictor["premium_workflow_id"] = str(args.hardcase_premium_workflow_id)
        save_json(
            os.path.join(dirs["metrics"], "hardcase_predictor.json"),
            hardcase_predictor,
        )
        monitor.heartbeat(
            note=(
                "hardcase predictor fitted "
                f"(n={hardcase_predictor.get('n_samples', 0)}, "
                f"train_acc={hardcase_predictor.get('train_acc', 0.0):.3f}, "
                f"fail_rate={hardcase_predictor.get('label_fail_rate', 0.0):.3f})"
            )
        )
    thresholds = {
        "train_mbpp": difficulty_threshold(train_mbpp),
        "mbpp_eval": difficulty_threshold(test_mbpp),
        "humaneval_eval": difficulty_threshold(test_humaneval),
    }
    # Keep warmup/probing calls out of run metrics.
    reset_global_counters()
    reset_runtime_telemetry()
    sample_trace_path = (
        os.path.join(dirs["logs"], "sample_trace.jsonl")
        if (args.enable_sample_trace and args.mode != "masrouter")
        else None
    )

    optimizer = torch.optim.Adam(router.parameters(), lr=0.01)

    monitor.set_stage("train")
    train_metrics = {}
    for ep in range(args.epochs):
        monitor.heartbeat(note=f"epoch={ep+1}/{args.epochs}")
        m = train_or_eval_epoch(
            mode=args.mode,
            router=router,
            data=train_mbpp,
            tasks=tasks,
            reasonings=reasonings,
            optimizer=optimizer,
            monitor=monitor,
            prompt_file=prompt_file,
            batch_size=args.batch_size,
            cost_rate=args.cost_rate,
            exec_timeout_s=args.exec_timeout_s,
            batch_timeout_s=args.batch_timeout_s,
            force_workflow_id=force_workflow_id,
            split_name="train_mbpp",
            difficulty_thr=thresholds["train_mbpp"],
            routing_tracker=routing_tracker,
            workflow_call_budget=wf_call_budget,
            cheap_first_cheap_workflow_id=args.cheap_workflow_id,
            cheap_first_premium_workflow_id=args.premium_workflow_id,
            hardcase_predictor=hardcase_predictor,
            hardcase_tau=float(args.hardcase_tau),
            hardcase_cheap_workflow_id=args.cheap_workflow_id,
            hardcase_premium_workflow_id=args.hardcase_premium_workflow_id,
            sample_trace_path=sample_trace_path,
            run_id=run_id,
            workflow_entropy_reg=float(args.workflow_entropy_reg),
            train=True,
        )
        train_metrics[f"epoch_{ep+1}"] = m
        ckpt = os.path.join(dirs["checkpoints"], f"{args.mode}_epoch{ep+1}.pth")
        torch.save(router.state_dict(), ckpt)

    monitor.set_stage("eval_mbpp")
    mbpp_eval = train_or_eval_epoch(
        mode=args.mode,
        router=router,
        data=test_mbpp,
        tasks=tasks,
        reasonings=reasonings,
        optimizer=optimizer,
        monitor=monitor,
        prompt_file=prompt_file,
        batch_size=args.batch_size,
        cost_rate=args.cost_rate,
        exec_timeout_s=args.exec_timeout_s,
        batch_timeout_s=args.batch_timeout_s,
        force_workflow_id=force_workflow_id,
        split_name="mbpp_eval",
        difficulty_thr=thresholds["mbpp_eval"],
        routing_tracker=routing_tracker,
        workflow_call_budget=wf_call_budget,
        cheap_first_cheap_workflow_id=args.cheap_workflow_id,
        cheap_first_premium_workflow_id=args.premium_workflow_id,
        hardcase_predictor=hardcase_predictor,
        hardcase_tau=float(args.hardcase_tau),
        hardcase_cheap_workflow_id=args.cheap_workflow_id,
        hardcase_premium_workflow_id=args.hardcase_premium_workflow_id,
        sample_trace_path=sample_trace_path,
        run_id=run_id,
        workflow_entropy_reg=float(args.workflow_entropy_reg),
        train=False,
    )

    monitor.set_stage("eval_humaneval")
    humaneval_eval = train_or_eval_epoch(
        mode=args.mode,
        router=router,
        data=test_humaneval,
        tasks=tasks,
        reasonings=reasonings,
        optimizer=optimizer,
        monitor=monitor,
        prompt_file=prompt_file_humaneval,
        batch_size=args.batch_size,
        cost_rate=args.cost_rate,
        exec_timeout_s=args.exec_timeout_s,
        batch_timeout_s=args.batch_timeout_s,
        force_workflow_id=force_workflow_id,
        split_name="humaneval_eval",
        difficulty_thr=thresholds["humaneval_eval"],
        routing_tracker=routing_tracker,
        workflow_call_budget=wf_call_budget,
        cheap_first_cheap_workflow_id=args.cheap_workflow_id,
        cheap_first_premium_workflow_id=args.premium_workflow_id,
        hardcase_predictor=hardcase_predictor,
        hardcase_tau=float(args.hardcase_tau),
        hardcase_cheap_workflow_id=args.cheap_workflow_id,
        hardcase_premium_workflow_id=args.hardcase_premium_workflow_id,
        sample_trace_path=sample_trace_path,
        run_id=run_id,
        workflow_entropy_reg=float(args.workflow_entropy_reg),
        train=False,
    )

    monitor.set_stage("finalize")
    summary = {
        "run_id": run_id,
        "mode": args.mode,
        "seed": int(args.seed),
        "train_samples": args.train_samples,
        "calibration_size": args.calibration_size,
        "test_samples_mbpp": args.test_samples_mbpp,
        "test_samples_humaneval": args.test_samples_humaneval,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "cost_rate": args.cost_rate,
        "max_agent": args.max_agent,
        "calibration_mix": args.calibration_mix,
        "exec_timeout_s": args.exec_timeout_s,
        "batch_timeout_s": int(args.batch_timeout_s),
        "no_fallback": bool(args.no_fallback),
        "require_heterogeneous_endpoints": bool(args.require_heterogeneous_endpoints),
        "exclude_budget_tiers": excluded_tiers,
        "cheap_workflow_id": args.cheap_workflow_id,
        "premium_workflow_id": args.premium_workflow_id,
        "force_workflow_id": force_workflow_id,
        "model_endpoints": args.model_endpoints,
        "inject_tests_into_humaneval_query": bool(args.inject_tests_into_humaneval_query),
        "premium_require_tests": bool(int(args.premium_require_tests)),
        "roi_gate_margin": float(args.roi_gate_margin),
        "deterministic_inference": bool(args.deterministic_inference),
        "deterministic_router_components": bool(args.deterministic_router_components),
        "workflow_prior_beta": float(args.workflow_prior_beta),
        "workflow_entropy_reg": float(args.workflow_entropy_reg),
        "hardcase_tau": float(args.hardcase_tau),
        "hardcase_calibration_size": int(args.hardcase_calibration_size),
        "hardcase_premium_workflow_id": str(args.hardcase_premium_workflow_id),
        "premium_prior_epsilon": float(args.premium_prior_epsilon),
        "endpoint_gate": endpoint_gate,
        "endpoint_heterogeneity": hetero_summary,
        "mbpp_eval": mbpp_eval,
        "humaneval_eval": humaneval_eval,
    }

    save_json(os.path.join(dirs["metrics"], "train_metrics.json"), train_metrics)
    save_json(os.path.join(dirs["metrics"], "mbpp_eval.json"), mbpp_eval)
    save_json(os.path.join(dirs["metrics"], "humaneval_eval.json"), humaneval_eval)
    if pareto_json:
        summary["pareto_library"] = pareto_json
    if prior_report is not None:
        prior_report_path = os.path.join(dirs["metrics"], "prior_gating_report.json")
        save_json(prior_report_path, prior_report)
        summary["prior_gating_report"] = prior_report_path
    if hardcase_predictor is not None:
        summary["hardcase_predictor"] = os.path.join(
            dirs["metrics"], "hardcase_predictor.json"
        )
    if routing_tracker is not None:
        routing_json = os.path.join(dirs["metrics"], "routing_analysis.json")
        save_json(routing_json, routing_tracker.to_dict())
        summary["routing_analysis"] = routing_json
        if sample_trace_path:
            summary["sample_trace"] = sample_trace_path
        premium_debug_path = os.environ.get("WAE_PREMIUM_DEBUG_LOG", "").strip()
        if premium_debug_path:
            summary["premium_debug_log"] = premium_debug_path
    save_json(os.path.join(dirs["metrics"], "summary.json"), summary)

    report_path = os.path.join(dirs["root"], "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# WaE-Router Pilot Report\n\n")
        f.write(f"- run_id: `{run_id}`\n")
        f.write(f"- mode: `{args.mode}`\n")
        f.write(f"- mbpp pass@1: `{mbpp_eval['accuracy_or_pass1']:.4f}`\n")
        f.write(f"- humaneval pass@1: `{humaneval_eval['accuracy_or_pass1']:.4f}`\n")
        f.write(f"- mbpp avg cost: `{mbpp_eval['avg_cost']:.6f}`\n")
        f.write(f"- humaneval avg cost: `{humaneval_eval['avg_cost']:.6f}`\n")
        f.write(f"- latency p50/p95 (mbpp): `{mbpp_eval['latency_p50_s']:.3f}` / `{mbpp_eval['latency_p95_s']:.3f}`\n")
        f.write(f"- latency p50/p95 (humaneval): `{humaneval_eval['latency_p50_s']:.3f}` / `{humaneval_eval['latency_p95_s']:.3f}`\n")
        f.write(
            f"- latency breakdown p50 (mbpp) [router/llm/test]: "
            f"`{mbpp_eval['router_overhead_p50_s']:.3f}` / `{mbpp_eval['llm_infer_p50_s']:.3f}` / `{mbpp_eval['test_exec_p50_s']:.3f}`\n"
        )
        f.write(
            f"- latency breakdown p50 (humaneval) [router/llm/test]: "
            f"`{humaneval_eval['router_overhead_p50_s']:.3f}` / `{humaneval_eval['llm_infer_p50_s']:.3f}` / `{humaneval_eval['test_exec_p50_s']:.3f}`\n"
        )
        f.write(f"- endpoint ready(all): `{endpoint_gate['all_ready']}`\n")
        f.write(f"- endpoint heterogeneity(model/base_url): `{hetero_summary['num_unique_model_ids']}` / `{hetero_summary['num_unique_base_urls']}`\n")
        if force_workflow_id:
            f.write(f"- forced workflow: `{force_workflow_id}`\n")
        if pareto_json:
            f.write(f"- pareto library: `{pareto_json}`\n")
        if routing_tracker is not None:
            f.write("- routing analysis: `metrics/routing_analysis.json`\n")
            if summary.get("premium_debug_log"):
                f.write("- premium debug log: `logs/premium_debug.jsonl`\n")

    monitor.heartbeat(note="completed")
    monitor.set_stage("completed")
    logger.info(f"Completed run: {run_id}")


if __name__ == "__main__":
    main()
