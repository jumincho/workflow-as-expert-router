"""Workflow-as-Expert profile definitions for the WaE pilot."""

from __future__ import annotations

from typing import Dict, List


CODE_ROLES: List[str] = [
    "ProjectManager",
    "ReflectProgrammer",
    "BugFixer",
    "AlgorithmDesigner",
    "TestAnalyst",
    "PlanSolver",
    "ProgrammingExpert",
]

WORKFLOW_PROFILE: List[Dict[str, object]] = [
    {
        "id": "wf_io_general",
        "name": "IO-General",
        "description": "Single-pass generation for easy tasks.",
        "base_model": "local-general",
        "method": "io",
        "params": {},
        "allowed_roles": CODE_ROLES,
        "budget_tier": "cheap",
        "cost_profile": {"input": 0.20, "output": 0.20},
    },
    {
        "id": "wf_refine2_coder",
        "name": "Refine2-Coder",
        "description": "Draft and one self-revision pass for robust code answers.",
        "base_model": "local-coder",
        "method": "refine2",
        "params": {"rounds": 1},
        "allowed_roles": [
            "ProgrammingExpert",
            "AlgorithmDesigner",
            "BugFixer",
            "ReflectProgrammer",
            "TestAnalyst",
        ],
        "budget_tier": "balanced",
        "cost_profile": {"input": 0.24, "output": 0.24},
    },
    {
        "id": "wf_sc3_coder",
        "name": "SC3-Coder",
        "description": "Three-sample self-consistency with syntax-aware candidate filter.",
        "base_model": "local-coder",
        "method": "self_consistency3",
        "params": {"k": 3},
        "allowed_roles": [
            "ProgrammingExpert",
            "AlgorithmDesigner",
            "PlanSolver",
        ],
        "budget_tier": "premium",
        "cost_profile": {"input": 0.24, "output": 0.24},
    },
    {
        "id": "wf_gen3_test_select_coder",
        "name": "Gen3-TestSelect-Coder",
        "description": (
            "Generate up to 3 candidates and select a passing one using inline tests when available."
        ),
        "base_model": "local-coder",
        "method": "gen_test_select3",
        "params": {"k": 3, "exec_timeout_s": 8},
        "allowed_roles": [
            "ProgrammingExpert",
            "AlgorithmDesigner",
            "BugFixer",
            "TestAnalyst",
            "ReflectProgrammer",
        ],
        "budget_tier": "premium",
        "cost_profile": {"input": 0.24, "output": 0.24},
    },
    {
        "id": "wf_gen2_test_select_general",
        "name": "Gen2-TestSelect-General",
        "description": (
            "Generate up to 2 candidates and pick the best passing one with inline tests "
            "for cheaper hard-case handling."
        ),
        "base_model": "local-general",
        "method": "gen_test_select3",
        "params": {"k": 2, "exec_timeout_s": 8},
        "allowed_roles": [
            "ProgrammingExpert",
            "AlgorithmDesigner",
            "BugFixer",
            "TestAnalyst",
            "ReflectProgrammer",
        ],
        "budget_tier": "balanced",
        "cost_profile": {"input": 0.20, "output": 0.20},
    },
    {
        "id": "wf_gen3_test_select_general",
        "name": "Gen3-TestSelect-General",
        "description": (
            "Generate up to 3 candidates and select a passing one using inline tests when available "
            "on the stronger general model."
        ),
        "base_model": "local-general",
        "method": "gen_test_select3",
        "params": {"k": 3, "exec_timeout_s": 8},
        "allowed_roles": [
            "ProgrammingExpert",
            "AlgorithmDesigner",
            "BugFixer",
            "TestAnalyst",
            "ReflectProgrammer",
        ],
        "budget_tier": "premium",
        "cost_profile": {"input": 0.20, "output": 0.20},
    },
]


def workflow_map() -> Dict[str, Dict[str, object]]:
    """Build a dictionary keyed by workflow id."""
    return {wf["id"]: wf for wf in WORKFLOW_PROFILE}


def select_workflows_by_tier(tier: str) -> List[Dict[str, object]]:
    return [wf for wf in WORKFLOW_PROFILE if wf["budget_tier"] == tier]
