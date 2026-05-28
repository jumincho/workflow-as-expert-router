"""Catalog of candidate *workflows* the WaE router can choose between.

A workflow here is a *method* applied to a *base model* — not just a model
choice. The pilot's whole question was whether choosing this larger unit
(method + base) is a better routing decision than choosing a model alone.

Six workflows are registered, covering four methods on two local
endpoints (a general 7B model and a coder 7B model):

- `wf_io_general`            : one-shot answer on the general model.
- `wf_refine2_coder`         : answer-then-refine on the coder model.
- `wf_sc3_coder`             : 3-sample self-consistency on the coder model.
- `wf_gen3_test_select_coder`: candidate+compare with inline tests (k=3, coder).
- `wf_gen2_test_select_general`: cheaper candidate+compare on the general model (k=2).
- `wf_gen3_test_select_general`: candidate+compare with inline tests (k=3, general).

Each entry carries:

- `method`         — selects which `WorkflowLLM._run_*` branch executes
                     (`io`, `refine2`, `self_consistency3`, `gen_test_select3`,
                     plus `critique_refine` exposed via the wrapper).
- `base_model`     — endpoint name resolved through `EndpointManager`.
- `allowed_roles`  — MasRouter role names this workflow is eligible for.
- `budget_tier`    — `cheap` / `balanced` / `premium`; used to gate exclusion.
- `cost_profile`   — per-token prices fed into `MODEL_PRICE`.

`workflow_map()` returns these keyed by id, and
`select_workflows_by_tier()` filters by `budget_tier` for ablations.

See GLOSSARY.md → "The four candidate workflows" for plain-English names
and how the closure reports refer to them.
"""

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
