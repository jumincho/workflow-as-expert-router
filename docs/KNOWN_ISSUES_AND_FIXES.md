# Known Issues and Practical Fixes

## 1) Status stale -> rc=143 false stall

Observed previously in some dynamic runs:
- run already at `stage=completed`, but monitor still sees stale mtime and kills process.

Practical mitigation:
- Keep `CONTINUE_ON_ERROR=1` (already used) to avoid full round halt.
- Validate completion by presence of `metrics/summary.json` + `report.md`.
- For long offline steps, rely on actual run artifacts rather than status mtime only.

## 2) Connection error spikes during eval

Symptoms:
- log messages like `Error during execution of node ...: Connection error.`

Mitigation:
- Verify local vLLM endpoint health before resume.
- Keep retries enabled in orchestration.
- If endpoint instability repeats, restart vLLM and resume from missing queue.

## 3) Dynamic claim weaker than framework claim

Current pattern across rounds:
- WaE framework cost efficiency is strong.
- Dynamic routing additional gain vs strong baselines is unstable.

Implication:
- Separate conclusions into G1 (framework value) and G2 (routing value).

