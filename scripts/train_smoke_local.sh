#!/usr/bin/env bash
set -euo pipefail

echo "train_smoke_local.sh is deprecated; forwarding to scripts/train_stage1_smoke.sh" >&2
bash scripts/train_stage1_smoke.sh
