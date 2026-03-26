# Experiment Overview

## Problem Framing

We are testing whether **WaE-Router (Workflow-as-Expert)** is a meaningful extension over MasRouter:
- Experts are workflows (`LLM + method/protocol`) instead of bare LLM endpoints.
- Offline library + online routing is evaluated under cost/accuracy/latency tradeoffs.

## Round7r2 Matrix (Expected)

Total planned run count: 28

- Seed1:
  - `masrouter_balanced`, `masrouter_cheap`, `masrouter_premium`
  - `wae_static_cheap`, `wae_static_premium`
  - `wae_dynamic_no_premium`, `wae_dynamic`
  - `wae_dynamic_hardcase_gate`
  - `wae_cheap_first_escalate`
  - `wae_dynamic_control_forced_io_general`
  - `wae_dynamic_hardcase_gate_tau0p3`, `wae_dynamic_hardcase_gate_tau0p7`
- Seed2, Seed3:
  - `masrouter_balanced`
  - `wae_static_cheap`, `wae_static_premium`
  - `wae_dynamic_no_premium`, `wae_dynamic`
  - `wae_dynamic_hardcase_gate`
  - `wae_cheap_first_escalate`
  - `wae_dynamic_control_forced_io_general`

## Metrics

Core:
- MBPP / HumanEval `accuracy_or_pass1`
- `avg_cost`
- `latency_p50_s` and breakdown (`router_overhead`, `llm_infer`, `test_exec`)

Decision logic:
- dominance-first
- iso-cost band (`±5%`) with target improvement threshold

