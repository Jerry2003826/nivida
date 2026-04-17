# Artifact Naming Convention (PR1 ∩ PR2 shared reference)

**Status**: Draft — naming source-of-truth for PR1 (默认入口收敛) and PR2 (artifact hygiene).
**Scope**: data files, config files, adapter directories, submission / evaluation artifacts.
**Does not cover**: prompt/completion contract (see `docs/harness_alignment.md`).

PR1 和 PR2 任何涉及命名的决策都先与本文件对齐，避免两个 PR 在同一命名上冲突或产生中间态。如果本文件缺一项，先补本文件再动代码。

---

## 0. 四分类原则

| 分类 | 前缀 / 目录 | 是否进训练 pipeline | 生命周期 |
| --- | --- | --- | --- |
| **Canonical** | `stage{1,2,3}_*`, `official_*`, `synth_hard_triads*`, `baseline_eval.json`, `adapter_stage{1,2,3}_*/` | ✓ 是真实消费者 | 长期维护 |
| **Audit** | `audit_*` | ✗ 只读分析用 | 随上游重新生成 |
| **Smoke** | `smoke_*` 或 `smoke/` 子目录 | ✗ CI/本地冒烟 | 每次 smoke 覆盖 |
| **Archive** | `_archive/` 子目录 | ✗ 不再消费 | 保留用于溯源 |

**规则**：
- 任何不能一眼判断归属的文件都是 bug；补前缀或移入 `_archive/`。
- Canonical 产物的生成入口必须在 `scripts/train_stage*.sh` 或 `scripts/prepare_data.py` 中显式声明。
- Audit 和 Smoke 产物的消费者不能是训练 pipeline；若被引用则降级归类或重命名。

---

## 1. Canonical 清单（目标态）

### 1.1 `data/processed/`

| 文件 | 来源 | 消费者 |
| --- | --- | --- |
| `official_train_tagged.jsonl` | `scripts/prepare_data.py` w/ `data_official.yaml` | stage1/2/3 input |
| `official_test.jsonl` | 同上 | local replica eval |
| `stage1_format_align_train.jsonl` | `sft_dataset_builder --selection-profile stage1` | stage1 trainer |
| `stage1_format_align_valid.jsonl` | 同上（role=valid） | stage1 trainer |
| `stage2_distill_train.jsonl` | `sft_dataset_builder --selection-profile stage2` | stage2 trainer |
| `stage2_distill_valid.jsonl` | 同上 | stage2 trainer（teacher-solvable proxy） |
| `stage3_repair_train.jsonl` | `sft_dataset_builder --selection-profile stage3` | stage3 trainer |
| `stage3_repair_valid.jsonl` | 同上 | stage3 trainer |
| `baseline_eval.json` | `run_baseline` w/ full official input | stage3 `--repair-artifact`, replica eval |
| `teacher_benchmark.json` | `run_teacher_benchmark` w/ full official input | 分析报告 |
| `parsed_train.jsonl` | `prepare_data` stage 1 parse | `official_train_tagged` 的上游 |
| `global_rule_graph.json` | `run_rule_analysis` | teacher 诊断 |
| `hard_cases.json` | `hardcase_miner` | 分析 |

### 1.2 `data/synthetic/`

| 文件 | 来源 |
| --- | --- |
| `synth_hard_triads.jsonl` | `synth_generator --config configs/synth_hard_triads.yaml` |
| `synth_hard_triads_summary.json` | 同上 |

### 1.3 `data/splits/official/`

| 文件 |
| --- |
| `splits.json` |
| `split_report.json` |

### 1.4 `artifacts/`

| 目录 / 文件 |
| --- |
| `adapter_stage1_format/` |
| `adapter_stage2_distill/` ← 原 `adapter_stage2_selected_trace/` |
| `adapter_stage3_repair/` |
| `target_module_audit.json` |
| `submission_validation.json` |
| `chat_template_probe.json`（PR0 副产物） |
| `prompt_length_audit.json`（PR0 副产物） |

### 1.5 `configs/`

| 文件 | 用途 |
| --- | --- |
| `data_official.yaml` | 官方数据 prepare 入口（canonical 默认） |
| `data_sample.yaml` | sample fixture prepare 入口（新增） |
| `eval_official.yaml` | 官方 baseline/eval 入口（PR1 新增，替代旧 `eval.yaml`） |
| `eval_sample.yaml` | sample fixture eval 入口（新增） |
| `synth_hard_triads.yaml` | stage2 synth 配置 |
| `train_stage1_format.yaml` | stage1 trainer config |
| `train_stage2_distill.yaml` | stage2 trainer config ← 原 `train_stage2_selected_trace.yaml` |
| `train_stage3_repair.yaml` | stage3 trainer config |
| `train_stage2_distill_full.yaml` | H100 / Runpod 完整训练版 ← 原 `train_lora_official_full_runpod.yaml` |

### 1.6 `scripts/`

| 文件 |
| --- |
| `train_stage1_format_align.sh` |
| `train_stage2_distill.sh` |
| `train_stage3_repair.sh` |
| `train_stage2_distill_full.sh` ← 原 `train_official_full_runpod.sh` |
| `probe_chat_template.py`（PR0 辅助） |
| `audit_prompt_lengths.py`（PR0 辅助） |
| `inspect_target_modules.py` |
| `validate_submission.py` |
| `eval_competition_replica.py` |
| `build_selected_sft.py`（仅生成 audit 产物） |
| `build_repair_set.py`（仅生成 audit 产物） |

### 1.7 `data/processed/smoke/`（新增）

smoke 脚本独立目录，不污染上面的 canonical 文件：

```
data/processed/smoke/
  stage1_format_align_train.jsonl
  stage1_format_align_valid.jsonl
  stage2_distill_train.jsonl
  stage2_distill_valid.jsonl
  stage3_repair_train.jsonl
  stage3_repair_valid.jsonl
  baseline_eval.json
  synth/synth_hard_triads.jsonl
```

---

## 2. Audit 清单

Audit 产物只用于人工审核或回归分析，**不进训练 pipeline**。改前缀 `audit_*`，便于 grep/CI 检查。

| 当前文件 | 目标命名 | 内容语义 |
| --- | --- | --- |
| `data/processed/selected_train.jsonl` | `data/processed/audit_selected_stage2.jsonl` | `build_selected_sft.py` 的 180 条切片 |
| `data/processed/selected_report.json` | `data/processed/audit_selected_stage2_report.json` | 同上的摘要 |
| `data/processed/repair_train.jsonl` | `data/processed/audit_baseline_failures.jsonl` | `build_repair_set.py` 的 3076 条切片 |
| `data/processed/repair_report.json` | `data/processed/audit_baseline_failures_report.json` | 同上的摘要 |
| `data/processed/stage2_distill_train_synth_only.jsonl` | `data/processed/audit_stage2_synth_only.jsonl` | synth 部分的 audit 视图 |
| `data/processed/teacher_benchmark_equation_failures.json` | `data/processed/audit_teacher_equation_failures.json` | equation 家族专项分析 |
| `data/processed/teacher_benchmark_stage{1,2,3}.json` | `data/processed/audit_teacher_benchmark_stage{1,2,3}.json` | 各 stage 的 teacher 指标快照 |

---

## 3. Archive 清单（PR2 执行迁移）

以下文件是"之前迭代的产物"，不再被任何 pipeline 引用。统一迁到 `_archive/` 子目录保留历史，不进 git blame 盲区。

### 3.1 `data/processed/_archive/`

**Important correction (2026-04-18)**: the file names suggested that
`baseline_eval.json` was canonical and `baseline_eval_latest.json` was a stray
snapshot, but empirical verification during PR0 showed the opposite. The
current `baseline_eval.json` still follows a legacy schema without
`competition_correct`; the PR2 `RepairArtifactSchemaError` guard correctly
refuses it. The `_latest` file is the one that carries the canonical fields.

The migration below therefore **inverts** the earlier plan: promote
`baseline_eval_latest.json` to the canonical name and archive the old one.

| 来源 | 原因 |
| --- | --- |
| `data/processed/baseline_eval.json` (current) | Legacy schema without `competition_correct`; fails PR2 schema guard. Archive under a versioned name (e.g. `baseline_eval.pre_competition_correct.json`) and rename the `_latest` file to take its place. |
| `data/processed/teacher_benchmark.json` (current) | Same class of issue as baseline. If the `_latest` variant is the one the analysis tools consume, promote it to canonical; otherwise swap the direction. Verify before moving. |
| `data/processed/official_stage_c_train.jsonl` | "stage c" 旧命名，已被 stage2/3 取代 |
| `data/processed/official_stage_c_valid.jsonl` | 同上 |
| `data/processed/stage_c_train.jsonl` | 同上（样本级） |

### 3.2 `data/synthetic/_archive/`

| 来源 | 原因 |
| --- | --- |
| `data/synthetic/synth_stage_b.jsonl` | 旧 synth 产物 |
| `data/synthetic/synth_summary.json` | 对应旧 summary（和新 `synth_hard_triads_summary.json` 字段不同） |

### 3.3 `artifacts/_archive/`

| 来源 | 原因 |
| --- | --- |
| `artifacts/adapter/` | 最早期 adapter 雏形 |
| `artifacts/adapter_demo/` | 对应 `configs/train_lora.yaml` 的 demo 产物 |
| `artifacts/adapter_demo_submission.zip` | 同上 |
| `artifacts/adapter_official/` | 历史全量尝试 |
| `artifacts/adapter_official_full_runpod/` | 历史 Runpod 尝试 |
| `artifacts/runpod_downloads/` | 历史下载缓存 |
| `artifacts/baseline_eval_smoke.json` | smoke 遗留产物（smoke 路径要改到 `artifacts/smoke/`） |

### 3.4 `configs/_archive/` 或直接删除

| 来源 | 建议 | 原因 |
| --- | --- | --- |
| `configs/train_lora.yaml` | 归档 | `src/student/inference.py::main` 和 `src/student/lora_train.py::main` 的默认，需同步改默认到 `train_stage2_distill.yaml` |
| `configs/train_lora_official.yaml` | 删 | 未被任何脚本引用 |
| `configs/train_lora_smoke.yaml` | 改名迁 `configs/smoke/train_smoke.yaml` 或删除 | `scripts/train_smoke_local.sh` 要重写 |
| `configs/train_lora_stage3_repair.yaml` | 删 | 和 `configs/train_stage3_repair.yaml` 功能重复 |
| `configs/synth.yaml` | 归档 | 已被 `synth_hard_triads.yaml` 取代，但 `README.md` 还在引用 |

---

## 4. 迁移映射总表（PR1 + PR2 联合视图）

排列顺序：每一条 `现有名称 → 目标名称 | 执行 PR | 方式`。

### 4.1 Config 层（PR1）

| 现有 | 目标 | 执行方式 |
| --- | --- | --- |
| `configs/train_stage2_selected_trace.yaml` | `configs/train_stage2_distill.yaml` | `git mv` 保 blame |
| `configs/train_lora_official_full_runpod.yaml` | `configs/train_stage2_distill_full.yaml` | `git mv` |
| `configs/train_lora.yaml` | `configs/_archive/train_lora.yaml` | `git mv`，默认 config 指向改到 canonical |
| `configs/train_lora_official.yaml` | 删除 | `git rm` |
| `configs/train_lora_smoke.yaml` | `configs/smoke/train_smoke.yaml` | `git mv` |
| `configs/train_lora_stage3_repair.yaml` | 删除 | `git rm` |
| `configs/synth.yaml` | `configs/_archive/synth.yaml` | `git mv` |
| `configs/eval.yaml` | `configs/eval_sample.yaml` | `git mv`（内容不变），新建 `configs/eval_official.yaml` 指向官方数据 |
| `configs/data.yaml` | `configs/data_sample.yaml` | `git mv`（内容不变），`configs/data_official.yaml` 已存在作为 canonical |

### 4.2 Artifact / data 层（PR2）

| 现有 | 目标 | 方式 |
| --- | --- | --- |
| `data/processed/baseline_eval.json` (legacy schema) | `data/processed/_archive/baseline_eval.pre_competition_correct.json` | `git mv` |
| `data/processed/baseline_eval_latest.json` (canonical schema) | `data/processed/baseline_eval.json` | `git mv` (promote) |
| `data/processed/teacher_benchmark_latest.json` | pending verification — whichever variant carries the current schema wins the canonical name, the other goes to `_archive/` | verify then `git mv` |
| `data/processed/official_stage_c_*.jsonl` | `data/processed/_archive/official_stage_c_*.jsonl` | `git mv` |
| `data/processed/stage_c_train.jsonl` | `data/processed/_archive/stage_c_train.jsonl` | `git mv` |
| `data/processed/selected_train.jsonl` | `data/processed/audit_selected_stage2.jsonl` | `git mv` + 同步默认值 |
| `data/processed/selected_report.json` | `data/processed/audit_selected_stage2_report.json` | `git mv` + 同步 |
| `data/processed/repair_train.jsonl` | `data/processed/audit_baseline_failures.jsonl` | `git mv` + 同步 |
| `data/processed/repair_report.json` | `data/processed/audit_baseline_failures_report.json` | `git mv` + 同步 |
| `data/processed/stage2_distill_train_synth_only.jsonl` | `data/processed/audit_stage2_synth_only.jsonl` | `git mv` |
| `data/processed/teacher_benchmark_equation_failures.json` | `data/processed/audit_teacher_equation_failures.json` | `git mv` |
| `data/processed/teacher_benchmark_stage{1,2,3}.json` | `data/processed/audit_teacher_benchmark_stage{1,2,3}.json` | `git mv` |
| `data/synthetic/synth_stage_b.jsonl` | `data/synthetic/_archive/synth_stage_b.jsonl` | `git mv` |
| `data/synthetic/synth_summary.json` | `data/synthetic/_archive/synth_summary.json` | `git mv` |
| `artifacts/adapter/` | `artifacts/_archive/adapter/` | `git mv` |
| `artifacts/adapter_demo/`、`adapter_demo_submission.zip` | `artifacts/_archive/adapter_demo/` | `git mv` |
| `artifacts/adapter_official*/` | `artifacts/_archive/adapter_official*/` | `git mv` |
| `artifacts/adapter_smoke/` | `artifacts/smoke/adapter/` | `git mv` |
| `artifacts/adapter_stage2_selected_trace/` | `artifacts/adapter_stage2_distill/` | `git mv` |
| `artifacts/runpod_downloads/` | `artifacts/_archive/runpod_downloads/` | `git mv` |
| `artifacts/baseline_eval_smoke.json` | `artifacts/smoke/baseline_eval.json` | `git mv` |

---

## 5. 代码落点清单（改名时必须同步更新的默认值）

grep 结果完整清单。任何 PR 改到下表的行号必须同时更新 `docs/artifact_naming.md`。

### 5.1 默认 config 引用

| 文件 | 行 | 当前默认 | 目标默认 |
| --- | --- | --- | --- |
| `scripts/validate_submission.py` | 19 | `configs/train_stage2_selected_trace.yaml` | `configs/train_stage2_distill.yaml` |
| `scripts/inspect_target_modules.py` | 17 | 同上 | 同上 |
| `src/student/audit_target_modules.py` | 37 | 同上 | 同上 |
| `src/student/lora_train.py` | 475 | `configs/train_lora.yaml` | `configs/train_stage2_distill.yaml` |
| `src/student/inference.py` | 132 | 同上 | 同上 |
| `src/teacher/synth_generator.py` | 334 | `configs/synth.yaml` | `configs/synth_hard_triads.yaml` |
| `src/experiments/run_synth_ablation.py` | 12 | 同上 | 同上 |

### 5.2 默认输出路径

| 文件 | 行 | 当前 | 目标 |
| --- | --- | --- | --- |
| `scripts/build_selected_sft.py` | 21 | `data/processed/selected_train.jsonl` | `data/processed/audit_selected_stage2.jsonl` |
| `scripts/build_repair_set.py` | 22 | `data/processed/repair_train.jsonl` | `data/processed/audit_baseline_failures.jsonl` |

### 5.3 Shell 脚本

| 文件 | 行 | 当前 | 目标 |
| --- | --- | --- | --- |
| `scripts/train_stage2_distill.sh` | 24 | `configs/train_stage2_selected_trace.yaml` | `configs/train_stage2_distill.yaml` |
| `scripts/train_smoke_local.sh` | 5 | `configs/synth.yaml` | `configs/smoke/train_smoke.yaml` 对应 synth（重写全文件） |
| `scripts/train_smoke_local.sh` | 7, 15 | `data/synthetic/stage2_synth.jsonl` | 走 `data/processed/smoke/synth/...` |
| `scripts/train_smoke_local.sh` | 12, 20 | `--split-name hard_triad` | `hard_triad_rule_novelty` |
| `scripts/train_smoke_local.sh` | 25 | `configs/train_lora_smoke.yaml` | `configs/smoke/train_smoke.yaml` |
| `scripts/train_official_full_runpod.sh` | 6, 9 | `configs/train_lora_official_full_runpod.yaml` | `configs/train_stage2_distill_full.yaml` |

### 5.4 Makefile

| 行 | 现状 | 目标 |
| --- | --- | --- |
| 9 | `--config configs/data.yaml` | `--config configs/data_sample.yaml`（或保留 `data.yaml` 但让它 point 到 official） |
| 12 | `--config configs/eval.yaml` | `--config configs/eval_sample.yaml` |
| 15 | `--config configs/data.yaml` | 同 9 |

### 5.5 README.md

| 行 | 现状 | 目标 |
| --- | --- | --- |
| 50 | `configs/synth.yaml` | `configs/synth_hard_triads.yaml` |
| 72 | `data/synthetic/stage2_synth.jsonl` | `data/synthetic/synth_hard_triads.jsonl` |
| 163 | `configs/train_stage2_selected_trace.yaml` | `configs/train_stage2_distill.yaml` |
| 170 | 同上 | 同上 |

---

## 6. Canonical baseline artifact 的唯一性（对齐 PR2 schema 守卫）

`src/student/sft_dataset_builder.py::build_repair_set` 已经在 PR2 引入 `RepairArtifactSchemaError`（参见 `tests/test_sft_dataset_builder.py`），它假设唯一的 canonical artifact 是 `data/processed/baseline_eval.json`。本文件的 Archive 策略同步这个假设：

- `baseline_eval_latest.json` 迁入 `_archive/`，不再被任何代码路径读取
- schema 守卫 raise 时提示的 remediation 命令（`python -m src.experiments.run_baseline ... --output data/processed/baseline_eval.json`）与本文件 §1.1 的 canonical 路径一致
- 未来若需要 "latest" 概念，应通过 symlink 或 `baseline_eval.json` 的 schema 自带 `schema_version` 字段，而非重复文件

---

## 7. PR 执行顺序（对齐 `docs/handover.md` §7.4）

```
PR0 (harness alignment)
  ↓
PR0.5 (synth hard-family whitelist)
  ↓
PR1 (默认入口 / README / smoke 脚本)  ─┐
  ‖                                    ├─ 共同依赖本文件
PR2 (artifact 迁移 / schema guard) ─┘
  ↓
PR-adapter
  ↓
PR-H100
```

PR1 和 PR2 **必须在同一天 merge**，避免出现"默认 config 名已改但 canonical 产物名还没改"（反之亦然）的中间态。
如果不能同日 merge，PR1 必须在所有引用旧名的位置保留 compatibility shim（例如默认 config 里 include 旧 yaml 的内容），直到 PR2 合入。

---

## 8. 保留的兼容层（明确声明）

只保留**代码层**的 alias，用户可见入口全部收敛到 canonical：

- `src/competition/split_builder.py` 生成的 `splits.json` 同时写 `hard_triad` (alias) 和 `hard_triad_rule_novelty` (canonical)；`rule_novelty` 同理。代码使用者可继续用旧名，**文档和脚本不能**。
- `sft_dataset_builder.py::_validate_repair_artifact` 接受 legacy `rows` 字段但发 `DeprecationWarning`。

除此之外：
- 旧 config 文件名 → 不保留兼容层（删或归档）
- 旧 data 产物文件名 → 不保留兼容层（`git mv` 一次性迁移）
- 旧 shell 脚本名 → `git mv`

---

## 9. 检测工具（新增 CI test）

PR1 合入时加一条静态检查，grep 整个仓库（除 `docs/_archive/` 和 `CHANGELOG.md` 外）是否还在引用下列旧名：

```
configs/synth\.yaml
stage2_synth\.jsonl
configs/train_lora\.yaml
configs/train_lora_smoke\.yaml
configs/train_lora_official
configs/train_lora_stage3_repair
configs/train_stage2_selected_trace
data/processed/baseline_eval_latest
data/processed/teacher_benchmark_latest
selected_train\.jsonl
repair_train\.jsonl
stage_c_
official_stage_c
stage2_distill_train_synth_only
adapter_stage2_selected_trace
```

命中即 CI fail。实现落在 `tests/test_naming_convention.py`（新增）。

---

## 附录 A — 快速回答 "这个文件应该叫什么"

| 情景 | 前缀 / 路径 |
| --- | --- |
| 进 `train_stage{1,2,3}*.sh` 的数据 | `stage{1,2,3}_*` |
| teacher 蒸馏 synth | `synth_hard_triads*` |
| baseline 产物、唯一 canonical | `baseline_eval.json` |
| 审计切片（不进训练） | `audit_*` |
| smoke 产物 | `data/processed/smoke/` 或 `artifacts/smoke/` |
| 历史迭代遗留 | `_archive/` |
| adapter 输出目录 | `artifacts/adapter_stage{1,2,3}_*` |
| H100 上的 probe / audit 副产物 | 放 `artifacts/` 根 |

---

## 附录 B — 与 `docs/harness_alignment.md` 的职责边界

| 话题 | 归属文档 |
| --- | --- |
| prompt 包装、`<think>` 段、guard 文案 | `docs/harness_alignment.md` |
| completion 格式（answer_only / short_trace / token_trace） | `docs/harness_alignment.md` |
| 文件名、目录、config 名、移动方案 | 本文件 |
| adapter 交付策略（单 adapter / continue train） | `docs/adapter_delivery.md`（未起草） |

两份文档都不涉及 LoRA 超参、epoch 数、lr schedule 之类的训练配置——那些仍在 `configs/train_*.yaml` 里，作为实验数据不是命名契约。
