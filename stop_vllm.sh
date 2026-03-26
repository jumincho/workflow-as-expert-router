#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for pid_file in "${ROOT_DIR}"/runs/vllm_*.pid; do
  [[ -f "${pid_file}" ]] || continue
  pid="$(cat "${pid_file}")"
  if kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" || true
  fi
  rm -f "${pid_file}"
done

echo "vLLM servers stopped."

