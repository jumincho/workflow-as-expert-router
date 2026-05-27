#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${WAE_RUNS_ROOT:-${WAE_ROUTER_PILOT_ROOT:-${REPO_ROOT}}/runs}"
MODE="dry-run"
PRUNE_CHECKPOINT_PATTERN="pilot6h_*"

if [[ "${1:-}" == "--execute" ]]; then
  MODE="execute"
fi

if [[ "${2:-}" == "--pattern" && -n "${3:-}" ]]; then
  PRUNE_CHECKPOINT_PATTERN="$3"
fi

run_rm() {
  local target="$1"
  if [[ "$MODE" == "execute" ]]; then
    rm -rf "$target"
  fi
  echo "[$MODE] remove: $target"
}

run_rm_file() {
  local target="$1"
  if [[ "$MODE" == "execute" ]]; then
    rm -f "$target"
  fi
  echo "[$MODE] remove-file: $target"
}

echo "cleanup mode=$MODE root=$ROOT_DIR checkpoint_pattern=$PRUNE_CHECKPOINT_PATTERN"
echo "disk-before:"
du -sh "$ROOT_DIR"

shopt -s nullglob

# 1) Remove smoke runs.
for d in "$ROOT_DIR"/smoke_*; do
  [[ -d "$d" ]] || continue
  run_rm "$d"
done

# 2) Remove stale PID files.
for pid_file in "$ROOT_DIR"/*.pid; do
  [[ -f "$pid_file" ]] || continue
  pid="$(tr -d ' \n\r\t' < "$pid_file" || true)"
  if [[ -z "$pid" ]]; then
    run_rm_file "$pid_file"
    continue
  fi
  if ! ps -p "$pid" > /dev/null 2>&1; then
    run_rm_file "$pid_file"
  else
    echo "[$MODE] keep-active-pid: $pid_file -> $pid"
  fi
done

# 3) Remove empty vLLM logs.
if [[ -d "$ROOT_DIR/vllm_logs" ]]; then
  while IFS= read -r f; do
    run_rm_file "$f"
  done < <(find "$ROOT_DIR/vllm_logs" -type f -size 0c -name '*.log' | sort)

  if [[ -z "$(find "$ROOT_DIR/vllm_logs" -type f -print -quit)" ]]; then
    run_rm "$ROOT_DIR/vllm_logs"
  fi
fi

# 4) Prune old round checkpoints (*.pth) by run name pattern.
for run_dir in "$ROOT_DIR"/$PRUNE_CHECKPOINT_PATTERN; do
  [[ -d "$run_dir/checkpoints" ]] || continue
  while IFS= read -r pth; do
    run_rm_file "$pth"
  done < <(find "$run_dir/checkpoints" -type f -name '*.pth' | sort)
done

echo "disk-after:"
du -sh "$ROOT_DIR"
echo "cleanup completed"
