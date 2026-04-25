#!/usr/bin/env bash
set -euo pipefail

# Thin compatibility wrapper. The Python entrypoint is cross-platform and is
# the canonical local command on Windows.
cd "${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
exec "${PYTHON:-python}" scripts/build_stage2_answer_focused_data.py "$@"
