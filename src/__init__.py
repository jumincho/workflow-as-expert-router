"""WaE-Router (Workflow-as-Expert Router) pilot package.

A dormant research pilot that tested whether routing entire *workflows*
(one-shot, refine, candidate+compare, critique+rewrite) — rather than just
choosing a model — is a stronger expert unit for LLM systems.

The pilot was built on top of the external MasRouter (`MAR`) framework,
which is not vendored here; this package patches the upstream router so
its "LLM" slot can be filled by a workflow rather than a single endpoint.

Two findings survived:

- Workflow-as-expert routing beat model-level routing at matched accuracy
  across multiple rounds.
- The stronger claim that *dynamic* workflow choice is the core driver did
  not close: a static "just use the one cheap workflow" baseline was
  surprisingly strong.

See `GLOSSARY.md` at the repo root for the internal vocabulary used
across modules and closure reports (round names, snapshot keys, env vars,
the cost-matched comparison, the four candidate workflows, etc.).
"""
