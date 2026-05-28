"""Lightweight in-process monitor + a standalone status watcher.

`PilotMonitor` lives inside `run_pilot.py` and writes a single
`logs/status.json` per run, refreshed on each stage change, every step, and
on every heartbeat. The status file is the only thing external observers
(the watcher in this module, the snapshots under `status/` and
`artifacts/snapshots/`) need to know whether the run is alive.

What's tracked:

- `stage`               : free-form label set by the runner (`train`,
                          `eval_mbpp`, `eval_humaneval`, `completed`, ...).
- `error_count`         : monotonic count of `record_error` calls.
- rolling 20-step window of latency / cost / utility averages.
- `gpu_snapshot`        : best-effort `nvidia-smi` output (returns
                          `"nvidia-smi unavailable"` outside a GPU host).
- `event` / `note`      : tag of the last write (`step`, `heartbeat`,
                          `stage_change`, `error`).

`watch_status(path, interval_sec)` is a tiny CLI that tails one of those
status files and prints a one-line summary on each poll; that is what gets
invoked indirectly via `python -m src.monitor --status_path ...`. The
snapshots under `status/` and `artifacts/snapshots/` are *records* of what
this file produced during round7r2; see GLOSSARY → "status/ snapshot format".
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def _safe_json_dump(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _gpu_snapshot() -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        return out.strip()
    except Exception:
        return "nvidia-smi unavailable"


class PilotMonitor:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.logs_dir = os.path.join(run_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        self.status_path = os.path.join(self.logs_dir, "status.json")
        self.heartbeat_path = os.path.join(self.logs_dir, "heartbeat.log")
        self.error_count = 0
        self.step_metrics = []
        self.current_stage = "init"
        self.last_update = time.time()
        self.write_status(extra={})

    def set_stage(self, stage: str) -> None:
        self.current_stage = stage
        self.write_status(extra={"event": "stage_change"})

    def record_step(self, latency_s: float, cost: float, utility: float) -> None:
        self.step_metrics.append(
            {"latency_s": float(latency_s), "cost": float(cost), "utility": float(utility)}
        )
        if len(self.step_metrics) > 20:
            self.step_metrics = self.step_metrics[-20:]
        self.write_status(extra={"event": "step"})

    def record_error(self, message: str) -> None:
        self.error_count += 1
        with open(self.heartbeat_path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.utcnow().isoformat()} ERROR {message}\n")
        self.write_status(extra={"event": "error", "message": message})

    def heartbeat(self, note: Optional[str] = None) -> None:
        stamp = datetime.utcnow().isoformat()
        with open(self.heartbeat_path, "a", encoding="utf-8") as f:
            if note:
                f.write(f"{stamp} {note}\n")
            else:
                f.write(f"{stamp}\n")
        self.write_status(extra={"event": "heartbeat", "note": note or ""})

    def write_status(self, extra: Dict) -> None:
        self.last_update = time.time()
        avg_latency = (
            sum(x["latency_s"] for x in self.step_metrics) / len(self.step_metrics)
            if self.step_metrics
            else 0.0
        )
        avg_cost = (
            sum(x["cost"] for x in self.step_metrics) / len(self.step_metrics)
            if self.step_metrics
            else 0.0
        )
        avg_utility = (
            sum(x["utility"] for x in self.step_metrics) / len(self.step_metrics)
            if self.step_metrics
            else 0.0
        )
        payload = {
            "updated_at": datetime.utcnow().isoformat(),
            "stage": self.current_stage,
            "error_count": self.error_count,
            "recent_window": len(self.step_metrics),
            "avg_latency_s_last20": avg_latency,
            "avg_cost_last20": avg_cost,
            "avg_utility_last20": avg_utility,
            "gpu_snapshot": _gpu_snapshot(),
            **extra,
        }
        _safe_json_dump(self.status_path, payload)


def watch_status(status_path: str, interval_sec: int = 120) -> None:
    status_file = Path(status_path)
    print(f"[monitor] watching {status_file}")
    while True:
        if status_file.exists():
            data = json.loads(status_file.read_text(encoding="utf-8"))
            print(
                f"{data.get('updated_at')} stage={data.get('stage')} "
                f"errors={data.get('error_count')} avg_lat={data.get('avg_latency_s_last20'):.3f} "
                f"avg_cost={data.get('avg_cost_last20'):.6f} avg_util={data.get('avg_utility_last20'):.6f}"
            )
        else:
            print("status.json not found yet")
        time.sleep(interval_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch pilot status file")
    parser.add_argument("--status_path", required=True, type=str)
    parser.add_argument("--interval_sec", default=120, type=int)
    args = parser.parse_args()
    watch_status(args.status_path, args.interval_sec)


if __name__ == "__main__":
    main()
