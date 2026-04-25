# Answer-Only Run 2026-04-25

This note captures the GPU work completed on the Blackwell server before it was shut down.

## Server

- SSH: `root@103.196.86.105 -p 17321`
- Repo: `/workspace/nivida_h200_run`
- Git head: `e9ae897` (`Rescore legacy hard-triad annotations`)
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`, 97.9 GB VRAM

## Completed Training

Command:

```bash
python -m src.student.lora_train \
  --config configs/train_stage2_official_balanced_answer_only.yaml \
  --force-train
```

Output adapter:

```text
/workspace/nivida_h200_run/artifacts/adapter_stage2_official_balanced_answer_only
```

Saved checkpoints:

```text
checkpoint-100
checkpoint-200
checkpoint-294
final adapter_stage2_official_balanced_answer_only
```

Observed training diagnostics:

| step | signal |
| ---: | --- |
| 25 | `LossDiag=0.7888`, finite gradients |
| 50 | `LossDiag=0.5458`, finite gradients |
| 100 | `LossDiag=0.3071`, `eval_loss=0.3282`, checkpoint saved |
| 200 | `LossDiag=0.4557`, `eval_loss=0.3213`, checkpoint saved |
| 294 | final checkpoint and final adapter saved |

Final Trainer summary:

```text
train_runtime=5242.9411
train_samples_per_second=0.894
train_steps_per_second=0.056
train_loss=6.197609311058407
epoch=1.0
```

Interpretation: the answer-only continuation did update LoRA weights and did not hit NaN/Inf. It must still be ranked by parsed exact eval before any Kaggle submission.

## Inference Attempts

HF inference smoke:

- Command path: `scripts/run_cloud_inference_only_v3.sh`
- Input: `smoke_6pf`
- Candidates: `b_thin`, `answer_final`
- Cap: `MAX_NEW_TOKENS=512`
- Result: stopped manually after model load and several minutes of generation with no output file.
- Warning seen: `NemotronH requires an initialized NemotronHHybridDynamicCache...`

Conclusion: HF/PEFT generation is too slow for full local proxy evaluation on this model unless the cache path is fixed. Do not use it for large candidate sweeps.

vLLM proxy smoke:

- Existing venv: `/workspace/venvs/nemotron_t241`
- Found `vllm==0.19.1`
- Initial failure: `vllm/_C.abi3.so` ABI mismatch against `torch 2.8.0+cu128`
- Metadata showed `vllm 0.19.1` requires `torch==2.10.0`
- Installed in the vLLM venv before shutdown:
  - `torch==2.10.0`
  - `torchaudio==2.10.0`
  - `torchvision==0.25.0`
  - `transformers==4.56.2`
  - `setuptools==80.9.0`
  - `numpy==2.2.6`
  - `fsspec==2026.2.0`

The machine was shut down before the repaired vLLM environment was re-tested.

## Next GPU Boot

First, verify that the persistent disk still has the trained adapter:

```bash
test -f /workspace/nivida_h200_run/artifacts/adapter_stage2_official_balanced_answer_only/adapter_model.safetensors
```

Then run the CPU-only vLLM preflight:

```bash
cd /workspace/nivida_h200_run
bash scripts/check_cloud_vllm_env.sh
```

If the preflight passes, run the smallest vLLM smoke:

```bash
head -n 6 data/processed/local_eval_manifests/smoke_6pf.jsonl \
  > data/processed/local_eval_manifests/smoke_head6.jsonl

EVAL_INPUTS=smoke_head6 \
ADAPTERS="answer_final=artifacts/adapter_stage2_official_balanced_answer_only" \
bash scripts/run_cloud_vllm_exact_eval_v3.sh
```

Only after that succeeds should we run the candidate ranking inputs:

```bash
EVAL_INPUTS=combined_balanced_48pf,proxy_all_balanced_64pf,hard_triad_full \
ADAPTERS="b_thin=artifacts/adapter_stage2_thin,official_balanced=artifacts/adapter_stage2_thin_official_balanced_20260424_161110Z,answer_final=artifacts/adapter_stage2_official_balanced_answer_only,answer_ckpt_100=artifacts/adapter_stage2_official_balanced_answer_only/checkpoint-100,answer_ckpt_200=artifacts/adapter_stage2_official_balanced_answer_only/checkpoint-200,answer_ckpt_294=artifacts/adapter_stage2_official_balanced_answer_only/checkpoint-294" \
bash scripts/run_cloud_vllm_exact_eval_v3.sh
```

Pull back only the prediction/eval artifacts, then shut the GPU down again.

Local scoring after pulling `data/processed/vllm_exact_eval_v3`:

```bash
python scripts/score_vllm_exact_eval_outputs.py \
  --predictions-root data/processed/vllm_exact_eval_v3 \
  --output-root data/processed/eval/vllm_exact_eval_v3
```
