"""Process-isolated Python test execution utilities for pilot stability."""

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

