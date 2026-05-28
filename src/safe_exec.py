"""Subprocess-isolated runner for candidate Python code against unit tests.

`gen_test_select3` (and the offline Pareto builder during calibration) runs
each candidate completion against the inline tests embedded in the prompt.
Doing that in-process would leak imports, mutate state, and let a runaway
candidate hang the whole pilot.

`run_python_tests(code, tests, timeout_s)` therefore:

- spins up a fresh `python -c` subprocess,
- prepends `from typing import *` so common type-hint sugar resolves,
- joins the candidate code and the test lines as one script,
- enforces a hard wall-clock timeout (returns `(False, "TIMEOUT")` on hit),
- returns `(True, "")` only on `returncode == 0`,
- otherwise returns `(False, <tail of stderr or stdout>)` for log triage.

This is the only execution path the pilot uses for "did the candidate
actually pass". It is deliberately conservative about isolation: a single
broken candidate can never break the rest of the run.
"""

from __future__ import annotations

import subprocess
import sys
from typing import List, Tuple


def run_python_tests(code: str, tests: List[str], timeout_s: int = 10) -> Tuple[bool, str]:
    """Execute candidate code + tests in a fresh Python process with timeout."""
    script = "from typing import *\n" + code + "\n\n" + "\n".join(tests) + "\n"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    if proc.returncode == 0:
        return True, ""
    stderr = (proc.stderr or "").strip()
    stdout = (proc.stdout or "").strip()
    reason = stderr[-500:] if stderr else stdout[-500:]
    return False, reason
