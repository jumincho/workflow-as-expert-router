#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT_DIR}/.venv/bin/activate"

if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "HUGGINGFACE_HUB_TOKEN is not set"
  exit 1
fi

export HF_HOME="${HF_HOME:-${ROOT_DIR}/.hf_cache}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
mkdir -p "${ROOT_DIR}/runs/vllm_logs"

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --api-key "${VLLM_API_KEY}" \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  > "${ROOT_DIR}/runs/vllm_logs/general.log" 2>&1 &
echo $! > "${ROOT_DIR}/runs/vllm_general.pid"

CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --port 8001 \
  --host 0.0.0.0 \
  --api-key "${VLLM_API_KEY}" \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  > "${ROOT_DIR}/runs/vllm_logs/coder.log" 2>&1 &
echo $! > "${ROOT_DIR}/runs/vllm_coder.pid"

echo "Launched 2 vLLM servers. PID files are in ${ROOT_DIR}/runs/."

