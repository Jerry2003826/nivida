# Local-Only Work Products

目标：本地负责分析、评估、决策；云服务器只负责模型 forward/generation/probe 计算。

## 已完成的本地产物

### 1. Selection Stability 总表

- Markdown: `SELECTION_STABILITY_SUMMARY.md`
- CSV: `data/processed/route_probe/selection_stability_summary.csv`
- 生成脚本: `tools/render_selection_stability_report.py`

核心结论：

- visible public 名义权重 `0.25`，raw-count share `0.5165`。
- raw mixed 删除 public 的 mean leverage = `1.83`，最高。
- source-normalized 后，public leverage 回落到 `0.87`。
- raw mixed vs source-normalized mixed 的 top8 overlap 仍有 `6.83/8`，所以不能把掉分全归因于 public overweight。

### 2. Safe Expert Intersection

- Markdown: `SAFE_EXPERT_INTERSECTIONS_no_public_source_norm.md`
- JSON: `data/processed/route_probe/safe_intersections_no_public_source_norm.json`
- 脚本: `tools/analyze_safe_expert_intersections.py`

结果：

- checked modules: `46`
- norm-route 有交集: `37`
- 过滤后建议模块: `9`
- 推荐 expert slots: `14`

这说明后续若做 shared variant，应只动少数高置信交集，不要每层强行 top8。

### 3. Data Family Analysis

- Markdown: `DATA_FAMILY_ANALYSIS.md`
- JSON: `data/processed/data_family_analysis.json`
- CSV: `data/processed/data_family_analysis_family_rows.csv`
- 脚本: `tools/analyze_data_families.py`

关键发现：

- official train 六类基本均衡：bit/gravity/unit/cipher/numeral/equation 都约 `16-17%`。
- stage2_distill_train 明显不均衡：
  - cipher `32.7%`
  - bit `26.2%`
  - gravity `15.8%`
  - unit `14.8%`
  - equation `9.9%`
  - numeral `0.7%`
- stage2_distill_valid 只覆盖 cipher/bit/equation，缺 gravity/unit/numeral，不适合作为全分布 checkpoint selector。

### 4. Exact Eval Harness

- 脚本: `tools/evaluate_predictions_exact.py`
- 自检报告: `LOCAL_EXACT_EVAL_teacher_completion_stage2_valid.md`
- 自检 JSON: `data/processed/eval/teacher_completion_stage2_valid_exact_eval.json`
- 自检 CSV: `data/processed/eval/teacher_completion_stage2_valid_exact_eval_records.csv`

用途：

云服务器只生成 prediction JSONL；拉回本地后用这个脚本算：

- official-style extracted answer verify
- local competition-correct
- local exact
- boxed-valid rate
- family/subtype/answer-kind breakdown

示例：

```bash
python tools/evaluate_predictions_exact.py \
  --predictions data/processed/local_eval_predictions_v3/b_thin_predictions.jsonl \
  --labels data/processed/proxy_all_family_valid.jsonl \
  --output-json data/processed/eval/b_thin_proxy_all_exact_eval.json \
  --output-csv data/processed/eval/b_thin_proxy_all_exact_eval_records.csv \
  --output-md LOCAL_EXACT_EVAL_b_thin_proxy_all.md
```

### 5. Cloud Compute-Only Runbook

- Runbook: `CLOUD_COMPUTE_ONLY_PLAN.md`
- Remote script: `tools/run_cloud_compute_only_v3.sh`
- Inference-only remote script: `tools/run_cloud_inference_only_v3.sh`
- Detailed probe plan: `NEXT_RUN_minimal_route_probe_v3.md`

远端只做：

- per-example prompt-only route probe
- optional public generation diagnostic
- optional inference prediction generation

远端不做：

- training
- report analysis
- Kaggle submission
- leaderboard polling

## 下一次开云服务器的最小目标

1. 同步三个脚本到 `/workspace/nivida_h200_run/scripts/`：
   - `probe_nemotron_expert_routes.py`
   - `mix_route_reports.py`
   - `run_cloud_compute_only_v3.sh`
2. 运行：

```bash
bash scripts/run_cloud_compute_only_v3.sh
```

3. 拉回：

```bash
data/processed/route_probe_v3
```

4. 本地跑 stability 和 intersection。
5. 如果要评估模型输出，再用 `RUN_INFERENCE=1` 生成 predictions，拉回本地用 exact harness 评分。

## 新增：本地评估集和对比工具

### Local Eval Manifests

- 目录: `data/processed/local_eval_manifests`
- README: `data/processed/local_eval_manifests/README.md`
- 脚本: `tools/build_local_eval_manifests.py`

这些 manifest 是给云端 inference 用的 labeled JSONL。建议第一轮用：

```text
data/processed/local_eval_manifests/proxy_all_balanced_32pf.jsonl
```

它比 full proxy 小，能先快速看 family-level 是否有系统性伤害。

### Exact Eval Comparison

- 脚本: `tools/compare_exact_eval_reports.py`
- smoke 输出: `EXACT_EVAL_COMPARISON_smoke.md`

用法示例：

```bash
python tools/compare_exact_eval_reports.py \
  --baseline b_thin \
  --report b_thin=data/processed/eval/b_thin_proxy32_exact_eval.json \
  --report norm_shared=data/processed/eval/norm_shared_proxy32_exact_eval.json \
  --report raw_route=data/processed/eval/raw_route_proxy32_exact_eval.json \
  --output-md EXACT_EVAL_COMPARISON_proxy32.md \
  --output-csv data/processed/eval/exact_eval_comparison_proxy32.csv
```

### Safe Shared Candidate Plan

- Markdown: `SAFE_SHARED_CANDIDATE_PLAN_scale025.md`
- JSON: `data/processed/route_probe/safe_shared_candidate_plan_scale025.json`
- 脚本: `tools/export_safe_shared_candidate_plan.py`

这是一个低 scale、小交集的候选计划，不是提交建议。只有在 v3 per-example route 和 local exact eval 都支持时，才考虑构建 adapter。

## 当前不建议做的事

- 不建议直接训练。
- 不建议直接 build normalized route-shared adapter 并提交。
- 不建议把 visible public 放进 selection。
- 不建议每层固定 top8 注入。

当前优先级是：先证明 no-public per-example route 与 norm selection 有稳定交集，再用本地 exact eval 看是否有 family-level 系统性伤害。
