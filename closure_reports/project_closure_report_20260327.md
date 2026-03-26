# WaE-Router E2E Closure Report

Date: 2026-03-27  
Repository archived from: `/workspace/wae-router-e2e-handoff`

## 1. Executive Summary

This project tested whether **WaE-Router (Workflow-as-Expert)** is a practically meaningful extension over **MasRouter** for code-generation tasks.

The work pursued two related but distinct claims:

1. **G1: Framework claim**  
   Replacing bare-model experts with workflow experts should improve the cost/accuracy/latency tradeoff.
2. **G2: Dynamic routing claim**  
   Dynamic workflow selection should outperform strong WaE baselines such as `wae_static_cheap`, `wae_cheap_first_escalate`, and `wae_dynamic_control_forced_io_general`.

The most defensible final reading is:

- **G1 looks repeatedly supported.** Across multiple rounds, WaE-style workflow experts consistently showed strong cost-efficiency relative to MasRouter baselines.
- **G2 is not convincingly established.** Dynamic routing did not cleanly and repeatedly beat the strongest WaE baselines under dominance-first or iso-cost decision rules.
- The project therefore produced a useful systems result, but not the stronger routing-specific result originally hoped for.

This archive should be treated as a **historical handoff / closure bundle**, not as an active research codebase.

## 2. What This Project Was

The core idea was to route among **workflows** rather than only among raw LLM endpoints.

Instead of choosing only which model to call, the system defines reusable workflow experts such as:

- cheap direct I/O generation
- refine/coder workflows
- premium multi-candidate workflows
- critique/refine-style workflows

These workflows are wrapped to look like experts, then selected by an offline-plus-online routing stack. The experimental question was whether this extra structure produces a better operating frontier than MasRouter, and whether dynamic selection among workflows creates additional gains over simple static WaE baselines.

The main evaluation tasks were:

- `MBPP`
- `HumanEval`

The main metrics were:

- `accuracy_or_pass1`
- `avg_cost`
- `latency_p50_s` and `latency_p95_s`
- routing and workflow usage diagnostics

Decision logic emphasized:

- dominance-first comparisons
- iso-cost comparison bands
- explicit comparison against strong static and escalation baselines

## 3. Main Code and Repository Structure

The most important code lives in:

- `src/run_pilot.py`: main experiment runner and orchestration entry point
- `src/workflow_router_patch.py`: workflow-aware routing implementation
- `src/workflow_llm.py`: workflow wrapper that exposes workflows through an LLM-like interface
- `src/workflow_profile.py`: workflow definitions and profiles
- `src/offline_pareto_builder.py`: offline library / Pareto construction
- `src/compare_runs.py`: comparison logic, iso-cost evaluation, and reporting
- `src/monitor.py`: experiment monitoring

The operational package also includes:

- `config/`: endpoint and experiment configs
- `scripts/`: round execution and resume helpers
- `docs/`: runbook, experiment overview, known issues
- `status/`: last human-readable state snapshot
- `artifacts/`: past round reports and partial outputs

## 4. How The Project Evolved

### Early pilot stage

The earliest pilot asked whether the WaE idea showed any end-to-end signal at all.

Result:

- there was a promising signal on `MBPP`
- the signal did not cleanly generalize to `HumanEval`
- early runs still suffered from serving and scale limitations

This was enough to justify continuing, but not enough for a strong claim.

### Round5q1

Round5q1 was the first clearer decision-oriented round. It incorporated stronger baselines, dominance-first comparisons, and better routing analysis.

Main outcome:

- `wae_dynamic` showed a strong efficiency signal relative to MasRouter
- but it **failed the core routing claim** because it did not beat the strongest WaE baselines
- especially important: `wae_static_cheap` and `wae_cheap_first_escalate` remained very strong

Interpretation:

- the WaE framework looked useful
- the benefit seemed to come more from the framework and workflow design than from dynamic routing itself

### Round6r1

Round6r1 scaled the setup to a more serious multi-seed evaluation.

This is the round that most clearly stabilized the story:

- `wae_dynamic` vs `masrouter_balanced` 3-seed averages were favorable
  - `MBPP`: accuracy `+0.0083p`, cost `-42.7%`, p50 latency `-36.9%`
  - `HumanEval`: accuracy `+0.0625p`, cost `-54.3%`, p50 latency `-2.0%`
- however, the dynamic-routing claim still did not pass against strong WaE baselines
- compare verdicts failed across seeds for the routing-specific thesis

This was the strongest support for the final conclusion:

- **WaE as a system idea looks real**
- **dynamic routing as the key differentiator still looks weak or unstable**

### Round7r1

Round7r1 introduced additional work to strengthen the routing story:

- hardcase-gated variants
- richer tracing
- latency decomposition
- pre-analysis of possible headroom for routing improvements

But the round did not fully complete.

Only the Stage A reproducibility gate ran, and the full experimental matrix did not execute. The gate itself failed for the tested modes because token drift and some mismatches remained above target thresholds.

This means Round7r1 was useful for tooling and diagnosis, but not for a final claim.

### Round7r2

Round7r2 is the last recorded active round in this handoff package.

Planned matrix:

- `28` runs total

Status at handoff:

- `19` completed
- `9` missing / not yet started

The strongest partial evidence comes from seed-1 comparison artifacts. Those again suggest:

- dynamic routing does not cleanly win
- strong baselines remain highly competitive
- `wae_dynamic_control_forced_io_general` and `wae_static_cheap` are especially important controls

One especially important partial finding:

- on `HumanEval`, `wae_dynamic` was dominated by `wae_dynamic_control_forced_io_general`

That is not what one would want if the key story were “dynamic routing is the important new ingredient.”

## 5. What Was Actually Learned

### Strongest supported claim

The strongest claim this project supports is:

> Using workflows as experts appears to be a real systems improvement over a MasRouter-style setup in cost-efficiency terms.

This claim is supported by:

- pilot evidence
- stronger quick-core comparisons
- multi-seed evidence in Round6r1
- the general pattern that WaE variants are often cheaper while staying competitive or better on accuracy

### Weaker or unsupported claim

The weaker claim, and the one that never became convincing, is:

> Dynamic workflow routing itself is the main source of the gain.

Why this remained unsupported:

- `wae_dynamic` repeatedly failed to beat strong WaE baselines
- static cheap and cheap-first-escalate baselines were often excellent
- control modes suggested some gains were attributable to workflow/frame design, not selective dynamic routing
- partial Round7r2 evidence continued to weaken the routing-specific story

### Practical interpretation

If someone asked, “What should I believe after reading this repository?” the best answer is:

- **Believe in WaE as a useful engineering pattern.**
- **Do not overclaim that dynamic routing was proven to be the decisive ingredient.**

## 6. Why The Project Was Not Carried Forward

This codebase is being archived because the outcome no longer justifies active continuation in its current form.

Reasons:

1. The framework claim and routing claim diverged.
2. The framework claim is interesting, but narrower than the original ambition.
3. The routing claim did not mature into a clean, repeatable win over strong baselines.
4. The final recorded round remained incomplete.

In other words, the project generated real knowledge, but not a strong enough clean result to justify maintaining the entire experimental line as-is.

## 7. Final Status At Closure

At closure time, the honest status is:

- the repository is a small, well-structured handoff package
- the last active round is incomplete
- the strongest stable conclusion is a **framework-level** one, not a **routing-dominance** one
- the code is still useful for historical reference and possible future reuse
- the project should be considered **closed unless someone wants to revive it under a narrower systems framing**

## 8. If Someone Ever Reopens This Work

A future owner should not restart from the assumption that “dynamic routing will probably win with a little more tuning.”

A more realistic reopening strategy would be:

1. Frame the work primarily as a **workflow-expert systems efficiency** story.
2. Treat dynamic routing as a secondary or conditional hypothesis.
3. Require stronger ablations against:
   - `wae_static_cheap`
   - `wae_cheap_first_escalate`
   - forced-control workflow variants
4. Prefer larger multi-seed comparisons before claiming routing-specific value.

## 9. What Is Preserved In This Archive

Because the repository is already small, the closure bundle preserves almost the full handoff package, excluding `.git`.

It includes:

- source code
- configs
- orchestration scripts
- docs and runbooks
- status snapshots
- historical reports and partial artifacts
- this closure report in English and Korean

## 10. Bottom Line

This was a serious attempt to show that workflow-level routing beats plain model routing, and that dynamic workflow selection beats strong static WaE alternatives.

The project did uncover a useful result:

- **WaE looks like a meaningful systems pattern.**

But it did not cleanly prove the stronger result:

- **dynamic workflow routing as the decisive advantage remains unproven.**

That is the clearest and most newcomer-friendly reading of the repository at closure.
