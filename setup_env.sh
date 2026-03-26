#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

python3 -m venv "${VENV_DIR}" --system-site-packages
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${ROOT_DIR}/requirements.txt"
python -m pip install "vllm>=0.6.4"

python - <<'PY'
import importlib
mods = [
    "torch",
    "datasets",
    "sentence_transformers",
    "openai",
    "vllm",
    "class_registry",
    "shortuuid",
    "loguru",
    "tiktoken",
]
for m in mods:
    importlib.import_module(m)
print("import smoke passed")
PY

echo "Environment setup complete: ${VENV_DIR}"

