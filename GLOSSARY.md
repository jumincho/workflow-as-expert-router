# Glossary

The source tree and the closure reports use a few internal terms that aren't
self-explanatory if you're coming in cold. This is the decoder ring.

## The four candidate workflows (workflow-as-expert routing unit)

The pilot's entire premise is that the unit of routing should be a *workflow*
(method + base model), not a model. The four candidate methods correspond to
the categories in the closure reports:

| Closure-report name | Method id (in `src/workflow_profile.py`) | What it does |
|---|---|---|
| **one-shot** | `io` | Single forward call against the base model. The cheap, default workflow. Implemented in `WorkflowLLM._run_io`. |
| **answer-then-refine** | `refine2` | Draft a first answer, then a single self-revision pass. Implemented in `WorkflowLLM._run_refine2`. |
| **candidate-and-compare** | `gen_test_select3` (k=2 or k=3) | Generate `k` candidates, run any inline tests embedded in the prompt against each candidate, return the first one to pass (or the best by test-pass count if none does). Optionally repairs broken Python syntax once before scoring. Implemented in `WorkflowLLM._run_gen_test_select3`. |
| **critique-then-rewrite** | `critique_refine` | Draft, then critique the draft in a separate call, then rewrite using the critique. Implemented in `WorkflowLLM._run_critique_refine`. (Exposed in the wrapper but not selected as a Pareto candidate in the final rounds.) |

There is also `self_consistency3` — `k` samples, syntax-aware best-of —
which sits between *one-shot* and *candidate-and-compare* and is its own
profile entry (`wf_sc3_coder`).

The six concrete workflow profiles registered in `WORKFLOW_PROFILE`:

| Id | Method | Base model | Budget tier |
|---|---|---|---|
| `wf_io_general` | `io` | `local-general` | cheap |
| `wf_refine2_coder` | `refine2` | `local-coder` | balanced |
| `wf_sc3_coder` | `self_consistency3` | `local-coder` | premium |
| `wf_gen3_test_select_coder` | `gen_test_select3` (k=3) | `local-coder` | premium |
| `wf_gen2_test_select_general` | `gen_test_select3` (k=2) | `local-general` | balanced |
| `wf_gen3_test_select_general` | `gen_test_select3` (k=3) | `local-general` | premium |

## MasRouter (`MAR`) — the upstream router

| Term | What it means |
|---|---|
| **MasRouter** | The external multi-agent router this project builds on. It picks a task domain, a reasoning collab pattern, a set of roles, and an LLM per role. The WaE pilot replaces *the LLM choice* with a *workflow choice*. |
| **`MAR`** | The Python package name of MasRouter. Imports like `from MAR.MasRouter.mas_router import MasRouter` come from a separate checkout pointed to by `MASROUTER_PATH`. |
| **Why it's not vendored** | MasRouter has its own license, evolves independently, and ships large prompt/role JSONs that don't belong in this pilot. We patch `LLMRegistry.get` at runtime instead of forking the package. The `requirements.txt` here also deliberately does not install it. |

## Internal modules under `src/` — what each owns

| Module | What it owns |
|---|---|
| `workflow_profile` | The catalog of candidate workflows (id, method, base model, allowed roles, budget tier, cost profile). |
| `workflow_llm` | The wrapper that exposes a workflow as a single LLM-shaped callable (`gen` / `agen`). Also contains `EndpointLLM` — the OpenAI-compatible client for one local vLLM endpoint, with context-overflow retry and per-call telemetry. |
| `workflow_router_patch` | Three things: (a) `EndpointManager` loads endpoint configs and serves clients; (b) `LLMRuntimePatch` monkey-patches `LLMRegistry.get` so `wf::<id>` resolves to a `WorkflowLLM` while bare endpoint names still resolve to `EndpointLLM`; (c) `WaERouter` subclasses `MasRouter` and adds workflow-aware encoding + role-conditioned workflow selection (with optional `workflow_prior` mixing and `force_workflow_id`). |
| `offline_pareto_builder` | Offline cost-quality Pareto over the candidate workflows. Calibrates each workflow on an MBPP / HumanEval mini-set, drops dominated candidates, then splits the Pareto front by allowed-role to produce a `role_pareto_library.json` the router consumes. |
| `compare_runs` | Cost-matched comparison between baseline and target runs. Builds the baseline Pareto envelope on (cost, acc), runs iso-cost (tolerance band → linear interpolation → nearest fallback), runs dominance on (cost, acc) and (cost, acc, p50-latency), and applies the dominance-first +0.03 success rule. Emits JSON + Markdown + per-dataset scatter plot. |
| `analyze_results` | Cheap "did dynamic look better" smoke check that picks the latest run directories under `WAE_RUNS_ROOT` and reports nearest-cost delta. Lower fidelity than `compare_runs`. |
| `analyze_round_p0` | Per-round diagnostics: confusion table between `wae_dynamic` and `wae_static_cheap`, hard-case top-k where dynamic fails and cheap passes, dynamic-vs-forced-IO path diffs, dynamic-selection histograms, and oracle-headroom across static cheap/premium. |
| `monitor` | `PilotMonitor` writes `logs/status.json` and `heartbeat.log` inside each run dir as the runner progresses. The `watch_status` CLI tails one of those status files. |
| `run_pilot` | The actual experiment runner. Orchestrates endpoint warmup, MAR patch install, dataset loading, offline Pareto, optional training, evaluation on MBPP and HumanEval, and final summary/report writing. |
| `safe_exec` | Subprocess-isolated Python test runner with hard timeout. The only path used by `gen_test_select3` and the Pareto calibrator to ask "did this candidate code pass the inline tests". |

## Pilot modes (`--mode` on `run_pilot.py`)

| Mode | What it does |
|---|---|
| `masrouter` | Stock MasRouter baseline. Routes models, not workflows. |
| `wae_dynamic` | Workflow-as-expert dynamic routing. The main experimental arm. |
| `wae_static_cheap` | Force the cheap workflow everywhere. Baseline the headline-claim "dynamic beats static" had to beat. |
| `wae_static_premium` | Force the premium workflow everywhere. Upper-bound static comparison. |
| `wae_dynamic_no_premium` | Dynamic, but premium tier excluded from the candidate set. |
| `wae_dynamic_prior_gated` | Dynamic, but premium workflows are gated by their offline prior + ROI vs. cheap. |
| `wae_dynamic_roi_gated` | Dynamic, premium gated by ROI threshold only. |
| `wae_dynamic_hardcase_gate` | Heuristic gate: predict "will the cheap workflow fail?" with a logistic regressor over a few simple features, and only use premium on predicted hard cases. |
| `wae_cheap_first_escalate` | Always run cheap first; if its tests fail, escalate to premium for the unsolved subset. |

The `wae_dynamic_control_forced_io_general` label that appears in
`analyze_round_p0.py` is the same thing as `wae_dynamic` but with
`--force_workflow_id wf_io_general` — used as a control to check whether the
dynamic router's routing was actually choosing anything different from IO.

## Round names and snapshot artifacts

| Term | What it means |
|---|---|
| `roundNxK` (e.g. `round6r1`, `round7r1`, `round7r2`) | Internal experimental rounds. The trailing `rK` is the round-revision counter (re-runs of the same plan with fixes). `round7r2` was the last round attempted and did not complete (see `status/CURRENT_STATUS.md`). |
| `expanded_7b` | The 7B-model expansion experiment driven by `run_expanded_7b.sh` — same comparison plan as the main round flow but pinned to a fixed 7B endpoint set in `config/model_endpoints_3x7b.yaml`. |
| `status/` | The handoff snapshot folder. `CURRENT_STATUS.md` is the human-readable summary of where the last round was when it stopped; `ROUND7R2_PROGRESS.md` is the per-run table. `artifacts/snapshots/` carries the machine-readable JSON twins of the same state. |
| `status/<run>/logs/status.json` | Inside any run directory, the live status file that `monitor.PilotMonitor.write_status` keeps refreshed: `stage`, `error_count`, rolling 20-step averages of latency / cost / utility, `gpu_snapshot`. |

## Output directories — `closure_reports/` vs `artifacts/reports/`

| Directory | What it holds |
|---|---|
| `closure_reports/` | The **project-level closure reports** (KO + EN). One document per language, dated `2026-03-27`. These are the "what was the result, why is it dormant" writeups. The README links to them. |
| `artifacts/reports/` | Per-round / feedback-adoption reports. One file per round (`round5q1_report_ko_*`, `round6r1_report_ko_*`, `round7r1_report_ko_*`) plus the original `wae_router_round_report_*` summary. These are *intermediate* artifacts the closure reports synthesize. |
| `artifacts/round7r2/` | Partial outputs from the last (incomplete) round: per-seed compare JSON + Markdown for the dynamic mode and the hardcase-gate mode. |
| `artifacts/snapshots/` | Machine-readable JSON snapshots of round state — the runtime counterpart of the human-readable `status/` files. |

## Environment variables

| Variable | What it does |
|---|---|
| `WAE_ROUTER_PILOT_ROOT` | This repo's root. Used to resolve default config paths and the run-output directory. Defaults to `/workspace/wae_router_pilot` (the original author's host layout). |
| `MASROUTER_PATH` | Where the upstream MasRouter (`MAR`) checkout lives. The runner appends it to `sys.path` so `from MAR.MasRouter...` resolves. Defaults to `/workspace/masrouter`. |
| `WAE_RUNS_ROOT` | Where per-run output directories are created. Defaults to `${WAE_ROUTER_PILOT_ROOT}/runs`. |
| `WAE_MODEL_ENDPOINTS` | Path to the YAML config that lists the local vLLM endpoints and their per-token prices. Overridden per-run by `--model_endpoints`. |
| `WAE_EXPERIMENT_CONFIG` | Path to a YAML that can override command-line defaults. |
| `WAE_PREMIUM_DEBUG_LOG` | If set, `WorkflowLLM` writes one JSON record per candidate-and-compare invocation to this path. The runner sets it automatically per run when the mode isn't `masrouter`. |
| `WAE_PREMIUM_PHASE` / `WAE_PREMIUM_SPLIT` | Tagged into the premium debug log so calibration vs. eval vs. train cases can be separated. Set by the runner; not for users to touch. |
| `WAE_PREMIUM_REQUIRE_TESTS` | If `1`, `gen_test_select3` falls back to IO when no inline tests are parseable. Mirrors `--premium_require_tests`. |
| `WAE_PREMIUM_SYNTAX_REPAIR` | Controls the optional one-shot Python syntax-repair step inside `gen_test_select3` (default `1`). |
| `WAE_REQUEST_SEED`, `WAE_DEFAULT_TEMPERATURE`, `WAE_DEFAULT_TOP_P` | Force deterministic generation. Set automatically when `--deterministic_inference` is passed. |
| `WAE_ROUTER_GREEDY`, `WAE_ROUTER_GREEDY_ALL` | Force greedy argmax routing (workflow only, or all router components). Set automatically by `--deterministic_inference` / `--deterministic_router_components`. |
| `HUGGINGFACE_HUB_TOKEN` | Required for `launch_vllm.sh` to fetch the served models. |
| `VLLM_API_KEY` | API key shared between the vLLM server and the client (default `EMPTY` for local-only). |
| `HF_HOME` | Cache location for HuggingFace artifacts; `launch_vllm.sh` defaults it to `.hf_cache/` inside the repo. |
| `WAE_COMPARE_OUT_PREFIX` | Default output prefix for `compare_runs.py` JSON / MD / PNG outputs. |
| `WAE_ANALYSIS_OUTPUT` | Where `analyze_results.py` writes its summary JSON. |

## Benchmarks

| Name | What it is |
|---|---|
| **MBPP** | "Mostly Basic Python Problems". Hugging Face dataset `google-research-datasets/mbpp` ("sanitized" config). Each item has a natural-language prompt and a `test_list` of `assert` statements. Used for both train (`train` split) and primary evaluation (`test` split, key `mbpp_eval`). |
| **HumanEval** | OpenAI's HumanEval code-generation benchmark. Hugging Face dataset `openai_humaneval`. Each item has a function signature + docstring + a `check(...)` test runner over the named `entry_point`. Used as the second evaluation set (`humaneval_eval`). The `--inject_tests_into_humaneval_query` flag optionally appends the executable test block into the prompt so `gen_test_select3` can see it; without that, HumanEval is treated as no-inline-tests and falls back to IO. |

Both are code-generation pass@1 — a sample is "solved" iff its extracted Python
code passes the unit tests in `safe_exec.run_python_tests` within an enforced
subprocess timeout.

## Metrics

| Name | What it means |
|---|---|
| **`accuracy_or_pass1`** | Fraction of items in a split where the extracted code passes all inline tests. |
| **`avg_cost` / `total_cost`** | Sum and average of per-sample dollar cost, computed by MAR's `cost_count` against `MODEL_PRICE` per token. |
| **`latency_p50_s` / `latency_p95_s`** | End-to-end wall-clock latency per sample. |
| **Latency breakdown** (`router_overhead_p50_s` / `llm_infer_p50_s` / `test_exec_p50_s`) | Total latency split into (a) router overhead, (b) LLM inference, (c) test execution. The router overhead is computed as `total − llm_infer − test_exec` per sample. |
| **`call_count_est`** | Estimated number of LLM endpoint calls per sample. Used to detect when a workflow is doing more work than its cost implies. |
| **Iso-cost +0.03 success** | The headline pass/fail criterion in `compare_runs.py`: the target mode must (1) not be dominated by any baseline on (cost, acc) or (cost, acc, p50-lat), and (2) achieve ≥ +0.03 accuracy gap over the cost-matched baseline reference. |
| **Pareto envelope** | The non-dominated subset of baselines on (cost, acc). The iso-cost test runs against this envelope, not the full baseline set. |
| **Oracle headroom** | In `analyze_round_p0.py`: assuming a perfect picker between cheap and premium static workflows, how much would routing gain over cheap alone? Upper-bound for any dynamic policy. |

## Shell helpers — why these are separate from the Python flow

The four scripts at the repo root deliberately stay outside the Python flow
because they control infrastructure that the runner cannot control safely:

| Script | What it does |
|---|---|
| `setup_env.sh` | One-time project bootstrap. Creates `.venv/` with `--system-site-packages`, installs `requirements.txt`, installs vLLM, and runs an import smoke test for the heavyweight deps (`torch`, `vllm`, `datasets`, `openai`, ...). Run once per host. |
| `launch_vllm.sh` | Spawns three vLLM servers on `:8000` (Qwen2.5-7B-Instruct as `general`), `:8001` (Qwen2.5-Coder-7B-Instruct as `coder`), `:8002` (Qwen2.5-Math-7B-Instruct as `math`), each on a separate CUDA device, and writes PID files into `runs/`. Requires `HUGGINGFACE_HUB_TOKEN` for model fetch. This is the serving substrate — it lives outside Python because it's a long-running daemon set, not part of the experiment graph. |
| `stop_vllm.sh` | Reads the PID files `launch_vllm.sh` wrote and kills each server. |
| `run_expanded_7b.sh` | One canned 7B-pinned comparison sweep: masrouter / wae_dynamic / wae_static_cheap / wae_static_premium against `config/model_endpoints_3x7b.yaml`, followed by `compare_runs.py`. Separate from the per-round scripts under `scripts/` because the 7B set has different sample budgets and a different `--require_heterogeneous_endpoints` constraint. |

The Python entry points (`run_pilot.py`, `compare_runs.py`, etc.) all assume
the vLLM servers are already up. The shell scripts are the only thing that
brings them up.

## Why the round / version suffixes are kept in filenames

Closure reports, snapshot JSONs, and per-round comparison outputs cross-reference
each other by on-disk path (`artifacts/round7r2/round7r2_s1_compare_dynamic.md`,
`status/ROUND7R2_PROGRESS.md`, ...). Renaming the files would silently break
those cross-references and the historical record they constitute. The naming
convention is preserved as-is; this glossary is the bridge.
