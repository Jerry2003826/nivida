#!/usr/bin/env bash
set -euo pipefail

cd /workspace/nivida_remote

pid_file="/workspace/nivida_remote/train_stage2_distill_full_runpod.pid"
train_log="/workspace/nivida_remote/train_stage2_distill_full_runpod.log"
package_log="/workspace/nivida_remote/package_stage2_distill_full_runpod.log"
adapter_dir="artifacts/adapter_stage2_distill_full_runpod"
submission_zip="submission_stage2_distill_full_runpod.zip"

if [[ ! -f "${pid_file}" ]]; then
  echo "missing pid file: ${pid_file}" >&2
  exit 1
fi

pid="$(cat "${pid_file}")"
echo "watching training pid ${pid}" > "${package_log}"

while kill -0 "${pid}" 2>/dev/null; do
  sleep 60
done

if [[ ! -f "${adapter_dir}/adapter_model.safetensors" ]]; then
  echo "adapter artifact missing after training; packaging skipped" >> "${package_log}"
  tail -n 50 "${train_log}" >> "${package_log}" 2>/dev/null || true
  exit 1
fi

python -m src.student.package_submission --adapter-dir "${adapter_dir}" --output "${submission_zip}" >> "${package_log}" 2>&1
ls -lh "${submission_zip}" >> "${package_log}"
