# Handoff Runbook

## A. Preflight

1. Confirm VLLM endpoints are reachable:
- `http://127.0.0.1:8000/v1`
- `http://127.0.0.1:8001/v1`
- `http://127.0.0.1:8002/v1`

2. Confirm no stale process before restart:
```bash
pgrep -af 'run_round7_feedback.sh|python -m src.run_pilot|python -m src.compare_runs'
```

3. Confirm GPU health:
```bash
nvidia-smi
```

## B. Resume Round7r2

Run from `/workspace/wae_router_pilot` with existing `.venv`:
```bash
source .venv/bin/activate
PYTHONPATH=/workspace/wae_router_pilot:/workspace/masrouter \
ROUND_PREFIX=round7r2 \
RUN_STAGE_A=0 \
MAS_CURVE_ONLY_SEED1=1 \
HARDCASE_TAU_MAIN=0.5 \
HARDCASE_TAU_SWEEP='0.3 0.7' \
CONTINUE_ON_ERROR=1 \
RUN_MAX_RETRIES=1 \
./scripts/run_round7_feedback.sh
```

## C. Live Monitoring

```bash
tail -f /workspace/wae_router_pilot/runs/round7r2_orchestrator.log
```

Check one run status directly:
```bash
cat /workspace/wae_router_pilot/runs/<run_id>/logs/status.json
```

## D. Completion Criteria

- All expected run directories have `metrics/summary.json` + `report.md`
- Seed-level compare files exist (`round7r2_s*_compare_*.json/.md`)
- Final aggregated summary exists (`round7r2_summary.md`)

## E. Post-run Cleanup

```bash
bash /workspace/wae-router-round7-handoff/stop_vllm.sh
```

