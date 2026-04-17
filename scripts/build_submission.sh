#!/usr/bin/env bash
set -euo pipefail

python -m src.student.package_submission --adapter-dir artifacts/adapter --output submission.zip
