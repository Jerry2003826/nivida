# 下一次开机的最小路线：只做 probe / report，不训练不提交

目标：验证审核意见里的 `per-example normalized / no-public / prompt-only` route selection 是否稳定，并且只在有稳定交集时再考虑 shared variant。

## 原则

- 不直接训练。
- 不直接提交。
- 不把 visible public 放进 selection。
- visible public 只做 diagnostic。
- 优先 prompt-only route；generation route 只做单独诊断。
- probe 输出必须带 `--record-examples`，否则无法做 per-example normalized。

## 0. 服务器准备

把本地更新过的脚本同步到远端仓库：

```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 tools/probe_nemotron_expert_routes.py root@<HOST>:/workspace/nivida_h200_run/scripts/probe_nemotron_expert_routes.py
scp -P <PORT> -i ~/.ssh/id_ed25519 tools/probe_nemotron_expert_routes_batch.py root@<HOST>:/workspace/nivida_h200_run/scripts/probe_nemotron_expert_routes_batch.py
scp -P <PORT> -i ~/.ssh/id_ed25519 tools/mix_route_reports.py root@<HOST>:/workspace/nivida_h200_run/scripts/mix_route_reports.py
scp -P <PORT> -i ~/.ssh/id_ed25519 tools/analyze_route_selection_stability.py root@<HOST>:/workspace/nivida_h200_run/scripts/analyze_route_selection_stability.py
```

远端统一环境：

```bash
cd /workspace/nivida_h200_run
source /workspace/venvs/nemotron_t241/bin/activate
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH
export KAGGLEHUB_CACHE=/workspace/.cache/kagglehub
export HF_HOME=/workspace/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p data/processed/route_probe_v3 logs/setup_new_machine
```

## 1. no-public prompt-only per-example probe

推荐先用 batch probe，单次加载模型后连续跑所有 prompt-only job：

```bash
python scripts/probe_nemotron_expert_routes_batch.py \
  --config configs/train_stage2_thin.yaml \
  --adapter-dir artifacts/adapter_stage2_thin \
  --top-k 8 \
  --job "official_hard=data/processed/stage2_official_valid_hard_triad.jsonl:data/processed/route_probe_v3/official_hard_prompt_examples.json:256:prompt:0" \
  --job "official_all=data/processed/proxy_all_family_valid.jsonl:data/processed/route_probe_v3/official_all_prompt_examples.json:256:prompt:0" \
  --job "stage2_train=data/processed/stage2_distill_train.jsonl:data/processed/route_probe_v3/stage2_train_prompt_examples.json:512:prompt:0" \
  --job "public_visible=官方资料/test.csv:data/processed/route_probe_v3/public_visible_prompt_examples.json:35:prompt:0"
```

下面这些单独命令只作为 batch probe 失败时的 fallback。

```bash
python scripts/probe_nemotron_expert_routes.py \
  --config configs/train_stage2_thin.yaml \
  --input data/processed/stage2_official_valid_hard_triad.jsonl \
  --adapter-dir artifacts/adapter_stage2_thin \
  --output data/processed/route_probe_v3/official_hard_prompt_examples.json \
  --limit 256 \
  --top-k 8 \
  --count-scope prompt \
  --record-examples

python scripts/probe_nemotron_expert_routes.py \
  --config configs/train_stage2_thin.yaml \
  --input data/processed/proxy_all_family_valid.jsonl \
  --adapter-dir artifacts/adapter_stage2_thin \
  --output data/processed/route_probe_v3/official_all_prompt_examples.json \
  --limit 256 \
  --top-k 8 \
  --count-scope prompt \
  --record-examples

python scripts/probe_nemotron_expert_routes.py \
  --config configs/train_stage2_thin.yaml \
  --input data/processed/stage2_distill_train.jsonl \
  --adapter-dir artifacts/adapter_stage2_thin \
  --output data/processed/route_probe_v3/stage2_train_prompt_examples.json \
  --limit 512 \
  --top-k 8 \
  --count-scope prompt \
  --record-examples
```

## 2. visible public 只做 diagnostic

```bash
python scripts/probe_nemotron_expert_routes.py \
  --config configs/train_stage2_thin.yaml \
  --input 官方资料/test.csv \
  --adapter-dir artifacts/adapter_stage2_thin \
  --output data/processed/route_probe_v3/public_visible_prompt_examples.json \
  --limit 35 \
  --top-k 8 \
  --count-scope prompt \
  --record-examples
```

如果还要看 generation 轨迹，单独跑近似 generation-only diagnostic，不参与 selection：

```bash
python scripts/probe_nemotron_expert_routes.py \
  --config configs/train_stage2_thin.yaml \
  --input 官方资料/test.csv \
  --adapter-dir artifacts/adapter_stage2_thin \
  --output data/processed/route_probe_v3/public_visible_generation_delta_examples.json \
  --limit 35 \
  --top-k 8 \
  --max-new-tokens 192 \
  --count-scope generation_delta \
  --record-examples
```

## 3. 生成 per-example normalized no-public route report

```bash
python scripts/mix_route_reports.py \
  --normalization example \
  --input data/processed/route_probe_v3/official_hard_prompt_examples.json:0.40 \
  --input data/processed/route_probe_v3/official_all_prompt_examples.json:0.40 \
  --input data/processed/route_probe_v3/stage2_train_prompt_examples.json:0.20 \
  --output data/processed/route_probe_v3/mixed_example_norm_no_public_hard40_all40_train20.json
```

可选：生成带 public 的 diagnostic-only report，用来比 overlap/JSD，不用于 build adapter：

```bash
python scripts/mix_route_reports.py \
  --normalization example \
  --input data/processed/route_probe_v3/public_visible_prompt_examples.json:0.25 \
  --input data/processed/route_probe_v3/official_hard_prompt_examples.json:0.30 \
  --input data/processed/route_probe_v3/official_all_prompt_examples.json:0.30 \
  --input data/processed/route_probe_v3/stage2_train_prompt_examples.json:0.15 \
  --output data/processed/route_probe_v3/mixed_example_norm_with_public_diagnostic.json
```

## 4. 拉回本地分析

```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 -r root@<HOST>:/workspace/nivida_h200_run/data/processed/route_probe_v3 data/processed/
```

本地先看：

```bash
python tools/analyze_route_selection_stability.py \
  --route-dir data/processed/route_probe_v3 \
  --source public_visible=public_visible_prompt_examples.json:0.25 \
  --source official_hard=official_hard_prompt_examples.json:0.30 \
  --source official_all=official_all_prompt_examples.json:0.30 \
  --source stage2_train=stage2_train_prompt_examples.json:0.15 \
  --no-public-source official_hard=official_hard_prompt_examples.json:0.40 \
  --no-public-source official_all=official_all_prompt_examples.json:0.40 \
  --no-public-source stage2_train=stage2_train_prompt_examples.json:0.20 \
  --output data/processed/route_probe_v3/selection_stability_v3.json
```

## 5. 是否继续的判据

只有满足这些条件，才考虑开 GPU 做 inference eval 或低 scale shared variant：

- `S_per_example_no_public` 和 `S_norm_top8` 有稳定交集。
- 高 entropy / 低 top8_mass 层不动。
- 每层不强行 top8，只动交集里的 1-3 个高置信 expert。
- visible public leverage 高的 expert 不动。
- 先做 local parsed-exact inference eval，不直接提交 Kaggle。

建议的 shared variant 形态：

```text
experts = S_norm_top8 ∩ S_per_example_no_public_prompt_route
scale = 0.125 或 0.25
layers = high top8_mass / low entropy only
visible public = diagnostic only
```
