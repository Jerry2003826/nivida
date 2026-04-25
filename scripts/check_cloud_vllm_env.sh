#!/usr/bin/env bash
set -euo pipefail

# Run on the GPU server from /workspace/nivida_h200_run.
# This is a CPU-only preflight for the vLLM exact-eval environment. It does
# not instantiate the model and should show 0 MiB GPU usage when run alone.

cd "${REPO_DIR:-/workspace/nivida_h200_run}"

resolve_activate_path() {
  local candidate="$1"
  if [[ -d "$candidate" && -f "$candidate/bin/activate" ]]; then
    printf '%s\n' "$candidate/bin/activate"
    return 0
  fi
  if [[ -f "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  echo "Missing virtualenv activate script: $candidate" >&2
  echo "Set VENV to either a venv directory or its bin/activate file." >&2
  return 1
}

ACTIVATE_PATH="$(resolve_activate_path "${VENV:-/workspace/venvs/nemotron_t241/bin/activate}")"
VENV_ROOT="$(cd "$(dirname "$ACTIVATE_PATH")/.." && pwd)"
source "$ACTIVATE_PATH"

export LD_LIBRARY_PATH="${VENV_ROOT}/lib/python3.12/site-packages/nvidia/cusparselt/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
export KAGGLEHUB_CACHE="${KAGGLEHUB_CACHE:-/workspace/.cache/kagglehub}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

python - <<'PY'
from __future__ import annotations

import importlib.metadata as md
import re
import sys

import torch
import transformers
import vllm


def _requirement(name: str, needle: str) -> str:
    for req in md.metadata(name).get_all("Requires-Dist") or []:
        if needle.lower() in req.lower():
            return str(req)
    return ""


torch_req = _requirement("vllm", "torch")
torch_version = torch.__version__.split("+", 1)[0]
expected = ""
match = re.search(r"torch==([^; ]+)", torch_req)
if match:
    expected = match.group(1)

print(f"torch={torch.__version__} cuda={torch.version.cuda} file={torch.__file__}")
print(f"transformers={transformers.__version__}")
print(f"vllm={vllm.__version__} requires={torch_req or 'unknown'}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")

if expected and torch_version != expected:
    raise SystemExit(
        f"vLLM torch ABI mismatch: vllm requires torch=={expected}, "
        f"but imported torch {torch.__version__}"
    )
PY

nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader || true
