# 工作区交接审核包（Handover Review Packet）

**写给谁**：第一次接手这个仓库的工程师，希望在没有上下文的前提下完成"现状审核 + 给下一步建议"。
**读完需要**：≈30 分钟（快速 pass）；完整审核请按 §10 的路径展开。
**权威数据源位置**：`官方资料/`（已通过 Kaggle CLI 拉取，详见 §3）。

阅读顺序推荐：
1. §1 比赛硬约束 — 让你确定"什么是对，什么是错"
2. §2 仓库现状 — 告诉你"目前能跑到哪一步"
3. §4–§5 已识别的风险 — 审核重心
4. §7 悬而未决的决策 — 需要你拍板的地方
5. §10 速查清单 — 后续查证路径

---

## 1. 比赛硬约束（Ground-Truth Constraints）

比赛：**NVIDIA Nemotron Model Reasoning Challenge**（Kaggle）。
以下所有约束来自三个权威源：Kaggle 官方 demo notebook、官方 metric kernel、以及官方 overview
页面元数据。**凡与下表冲突的仓库实现都应视为 bug**。

### 1.1 Artifact 契约

| 项 | 值 | 来源 |
| --- | --- | --- |
| Base model | `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default` (kagglehub) | demo notebook cell 1 |
| Final artifact | 单个 LoRA adapter | demo / metric 一致 |
| LoRA rank 上限 | 32（严格） | `max_lora_rank=32` @ metric, demo `LORA_RANK = 32 # Can be set to a maximum of 32` |
| Submission 结构 | zip，内含 `adapter_config.json` + `adapter_model.safetensors|.bin` | demo cell 2 `zip -m submission.zip *` |
| Zip 内结构 | 扁平或单子目录皆可；harness `glob.glob('**/adapter_config.json', recursive=True)` 取第一个 | metric `generate_standard_submission` |
| 多 adapter 堆叠 | **不支持**（harness 只挂第一个 adapter_config.json） | metric `glob.glob(...)` 行为 |

### 1.2 推理契约

| 项 | 值 |
| --- | --- |
| 推理引擎 | **vLLM**（不是 transformers/peft；后者只在本地 dev 用） |
| `max_model_len` | 4096 |
| `max_tokens`（生成上限） | 3584 |
| **prompt 硬上限** | **4096 − 3584 = 512 BPE tokens** |
| `temperature` | 1.0 |
| `top_p` | 1.0 |
| `tensor_parallel_size` | 1（单卡） |
| `gpu_memory_utilization` | 0.85 |
| `enable_prefix_caching` | True |
| `enable_chunked_prefill` | True |
| `dtype` | `auto` |
| Runtime 环境变量 | `TRANSFORMERS_OFFLINE=1` — **不能联网下任何东西** |

### 1.3 Prompt 构造（harness 侧）

harness 对每一行 `test.csv` 做：

```python
user_content = item.prompt + '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
prompt = tokenizer.apply_chat_template(
    [{'role': 'user', 'content': user_content}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,   # Nemotron 的 thinking 模式
)
```

即：先拼 guard，再走 chat template + `enable_thinking=True`。
这意味着模型 **实际看到的 prompt 包含 role markers 和 thinking 开头 token**，
不是 `test.csv` 原文。

### 1.4 Answer 抽取与评分

抽取优先级（`extract_final_answer`）：

1. 所有 `\boxed{([^}]*)(?:\}|$)` 的 match 取**最后一个非空**
2. `The final answer is:` / `Final answer is:` / `Final answer:` / `final answer:`
3. 最后一个 `-?\d+(\.\d+)?` 数字
4. 最后一行
5. `'NOT_FOUND'`

比较（`verify`）：

- 二进制串（`^[01]+$`）：严格大小写不敏感串比较
- 否则尝试 `float()`，用 `rel_tol=1e-2, abs_tol=1e-5`
- 否则大小写不敏感串比较

### 1.5 数据集

| 文件 | 字段 | 来源 |
| --- | --- | --- |
| `train.csv` | `id, prompt, answer` | public, ~3 MB，`kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge` |
| `test.csv` | `id, prompt` | public, 35 行，本地打样用 |
| 真实评测数据 | `metric/nvidia-nemotron-rerun-data-129716` | private，仅 Kaggle runtime 内挂载 |

题目 6 个家族：`bit / cipher / equation / numeral / unit / gravity`；
其中 **hard triad = bit / cipher / equation**（codex_takeover_plan.md 的历史决策）。

### 1.6 提交配额和时间约束

- 每天 **5 次** 提交
- 最多 **2 个 final submission** 选为最终评分
- 单次 submission runtime 具体上限官方未在已拉取的 artifact 里暴露，按经验约 8–12 小时（未验证，需在 Kaggle 实际提交时确认）

---

## 2. 仓库现状

### 2.1 顶层结构

```
configs/      各阶段 yaml 配置
data/         processed（官方 + tagged）、splits、synthetic
docs/         设计文档（handover.md、harness_alignment.md）
notebooks/    3 个 EDA notebook（01/02/03）
scripts/      各阶段 shell + 辅助脚本
src/          competition / student / teacher / experiments / common
tests/        15 个 pytest 文件
artifacts/    dry-run 产物 + 多个历史 adapter_* 目录（见 §8.2）
官方资料/      Kaggle CLI 拉取的权威源（§3）
```

### 2.2 三阶段训练链路（设计意图）

| Stage | 训练目标 | 数据文件 | 完成行数 | completion 风格 | 入口脚本 |
| --- | --- | --- | --- | --- | --- |
| 1 format | 答案格式对齐 | `data/processed/stage1_format_align_train.jsonl` | 8077 | `answer_only` | `scripts/train_stage1_format_align.sh` |
| 1 format | — valid | `stage1_format_align_valid.jsonl` | 1423 | — | — |
| 2 distill | teacher 蒸馏（hard triad 为主） | `data/processed/stage2_distill_train.jsonl` | 4471 | `token_trace` | `scripts/train_stage2_distill.sh` |
| 2 distill | — valid | `stage2_distill_valid.jsonl` | 667 | — | — |
| 3 repair | baseline 失败样本修补 | `data/processed/stage3_repair_train.jsonl` | 2600 | `short_trace` | `scripts/train_stage3_repair.sh` |
| 3 repair | — valid | `stage3_repair_valid.jsonl` | 476 | — | — |

**重要区别**：
- stage1 只覆盖 official 样本（6 家族都有）
- stage2 = official 高置信样本 + `synth_hard_triads.jsonl` 合成样本（**含部分 easy 家族**，是 §4 风险 #3）
- stage3 = 以 `baseline_eval.json` 的失败样本子集构成，target_completion 带错误类型 bucket

### 2.3 Dry-run 状态

`pytest -q` 过：**81 passed**（最后一次运行）。
三个 stage 的 `artifacts/adapter_stage{1,2,3}_*/dry_run_manifest.json` 已生成。
**没有任何一条真实 GPU 训练发生过** —— 全部停在 dry-run 产物层。

### 2.4 已知需要外部验证的依赖

- `mamba_ssm`（Nemotron 必需）
- `causal-conv1d`
- `kagglehub`（拉 base model）
- `vllm`（headline eval 时使用）
- `cutlass`（从 Kaggle `/kaggle/usr/lib/notebooks/.../nvidia_cutlass_dsl/python_packages/` 挂载）

`src/student/lora_train.py` 的 `_maybe_add_kaggle_cutlass_path` 已经对齐 demo 的路径（demo notebook cell 1 用的就是这个 site.addsitedir）。

---

## 3. 已入库的权威数据源

通过 `kaggle` CLI 拉取到本地的副本：

| 路径 | 内容 | 重要度 |
| --- | --- | --- |
| `官方资料/kaggle_metric/nvidia-nemotron-metric.ipynb` | **官方 evaluator 本体**（vLLM + 抽取 + 比较） | 最高 |
| `官方资料/kaggle_demo/nvidia-nemotron-submission-demo.ipynb` | 官方 LoRA 训练 & 打包示例 | 高 |
| `官方资料/competition_data/unzipped/train.csv` | 3 MB public 训练集 | 高 |
| `官方资料/test.csv`（+ `competition_data/nvidia-nemotron-model-reasoning-challenge.zip`） | public test | 中 |
| `kaggle_overview.html` | 官方页面 HTML 快照（只含 meta，正文被 JS 渲染未抓到） | 低 |

拉取命令（Kaggle CLI 2.0.1，`~/.kaggle/kaggle.json` 已配置）：

```bash
kaggle kernels pull ryanholbrook/nvidia-nemotron-submission-demo -p 官方资料/kaggle_demo
kaggle kernels pull metric/nvidia-nemotron-metric                -p 官方资料/kaggle_metric
kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge -p 官方资料/competition_data
kaggle models instances versions list metric/nemotron-3-nano-30b-a3b-bf16/Transformers/default
```

如果权威源需要更新，直接覆盖下载即可。

---

## 4. 风险清单（六条仍在手 + 一条已消解）

严重度按 "对最终提交得分 / 可交付性 的影响" 定级。

| # | 问题 | 严重度 | 建议落点 |
| --- | --- | --- | --- |
| 1 | sample-fixture 默认入口群 | 中 | PR1 |
| 2 | `selected_train.jsonl` / `repair_train.jsonl` 旁路审计产物容易被误认为训练输入 | 低 | PR2 |
| 3 | stage2 synth 含 easy family（config + selection 双问题） | 中 | PR0.5 |
| 4 | 实现路径未在 H100 实测（inference / target_modules / length） | 高 | PR-H100 |
| 5 | Artifact hygiene（`*_latest.json` / 多套 `adapter_*` / `submission.zip` 1.5GB） | 中 | PR2 |
| 6 | `README.md` 和 `scripts/train_smoke_local.sh` 仍是旧口径 | 中 | PR1 |
| ✓ | LoRA `target_modules` regex 命中风险 | **已消解**（demo 亲自用同一正则） | — |

### 4.1 sample-fixture 默认入口群（#1）

- `configs/eval.yaml` line 1、`configs/data.yaml` line 1 都硬写 `tests/fixtures/sample_competition.csv`
- `scripts/prepare_data.py` line 21 默认 config 指向 `configs/data.yaml`
- `Makefile` line 8 / 12 的 `prepare` / `baseline` 调用这两个 yaml
- `src/experiments/run_baseline.py` line 48 里 config 优先级**覆盖** `--input` 命令行参数

**真正风险面**：有人直接跑 `make baseline` 会把 `data/processed/baseline_eval.json` 覆盖成 6 条 sample 结果；`scripts/train_stage3_repair.sh` 的 **line 4** 会在下游重跑 baseline 刷新全量，所以连贯执行不会被污染。被污染的场景是：**有人直接跑 stage3 构造而不先重刷 baseline**，吃到 sample 结果或旧 schema artifact。

### 4.2 旁路审计产物（#2）

| 产物 | 规模 | 是否进训练 | 命名建议 |
| --- | --- | --- | --- |
| `data/processed/selected_train.jsonl` | 180 | ❌ 不进 | → `audit_selected_*.jsonl` |
| `data/processed/repair_train.jsonl` | 3076 | ❌ 不进 | → `audit_repair_*.jsonl` |
| `data/processed/stage2_distill_train.jsonl` | 4471 | ✓ 是 stage2 真输入 | 保持 |
| `data/processed/stage3_repair_train.jsonl` | 2600 | ✓ 是 stage3 真输入 | 保持 |

### 4.3 stage2 synth 含 easy family（#3）

根源在两处：

1. `configs/synth_hard_triads.yaml` line 12–14 给 `numeral / unit / gravity` 都分了 weight = 0.4
2. `src/student/sft_dataset_builder.py::_select_synth_stage2`（line 200–210）没有 hard family 白名单

双保险做法：
- yaml 改成 0.0
- selection 加 `HARD_TRIAD_FAMILIES` 白名单（常量已在 `src/competition/split_builder.py` line 16）

### 4.4 实现路径未实测（#4）

已知未在真实 H100 + Nemotron 环境验证的路径：

- `src/student/inference.py` line 33 的 `_import_or_raise("mamba_ssm")`（demo 就是无条件 import，所以 Kaggle 场景 OK，但本地 smoke 走 TinyLlama 时炸）
- `configs/train_stage{1,2,3}*.yaml` 里的 `target_modules` regex（官方 demo 自用同一 regex，**风险已消解**）
- `src/student/lora_train.py::_simple_token_count` 是空格分词，dry-run manifest 的长度分位数不是真 BPE 长度

### 4.5 Artifact hygiene（#5）

`data/processed/` 和 `data/synthetic/` 里新旧产物并存：

| 文件 | 规模 | 状态 |
| --- | --- | --- |
| `baseline_eval.json` | 48 MB | 旧 schema |
| `baseline_eval_latest.json` | 38 MB | 新 schema |
| `teacher_benchmark.json` | 1.0 MB | 旧 |
| `teacher_benchmark_latest.json` | 1.3 MB | 新 |
| `teacher_benchmark_stage{1,2,3}.json` | ~40 KB each | 中间产物 |
| `synth_stage_b.jsonl` | 107 KB | 旧 synth |
| `official_stage_c_{train,valid}.jsonl` / `stage_c_train.jsonl` | 5 MB / 556 KB / 2 KB | "stage c" 旧命名 |
| `stage2_distill_train_synth_only.jsonl` | 2 MB | 审计切片 |

**风险路径**：`scripts/train_stage3_repair.sh` 默认消费 `baseline_eval.json`（旧），
`src/student/sft_dataset_builder.py::build_repair_set`（line 267–272）里
`failures = payload.get("records", payload.get("rows", []))` + `row.get("competition_correct", False)`
在旧 schema 无该字段时 **把所有样本当作失败样本处理**，静默吃进训练集。

### 4.6 README / smoke 脚本旧口径（#6）

| 文件 | 行 | 旧引用 | 应改 |
| --- | --- | --- | --- |
| `README.md` | 50 | `configs/synth.yaml` | `configs/synth_hard_triads.yaml` |
| `README.md` | 72 | `data/synthetic/stage2_synth.jsonl` | `data/synthetic/synth_hard_triads.jsonl` |
| `scripts/train_smoke_local.sh` | 5 | `configs/synth.yaml` | — |
| `scripts/train_smoke_local.sh` | 7 / 15 | `data/synthetic/stage2_synth.jsonl` | — |
| `scripts/train_smoke_local.sh` | 12 / 20 | split 别名 `hard_triad` | `hard_triad_rule_novelty` |

按 README 第 2/3 节照抄会产生 128 条旧 synth，覆盖校准过的 `stage2_distill_train.jsonl`。

---

## 5. 拿到 metric kernel 后的额外冲击（原计划未覆盖）

这一节是**最新发现**，驱动了 `docs/harness_alignment.md` 的设计。

### 5.1 Prompt / completion 契约和 harness 不一致

三条 misalignment，全部有 score impact：

- **Chat template 缺失**：仓库训练 prompt 不走 `apply_chat_template`，模型推理时才第一次见到 role markers 和 thinking 段 → 严重分布偏移
- **Guard 文案不同**：仓库 `ANSWER_CONTRACT = "Return exactly one final answer as \\boxed{...}."`；harness guard = `"\nPlease put your final answer inside \`\\boxed{}\`. For example: \`\\boxed{your answer}\`"`。弱分布差异但可消除
- **Thinking 段缺失**：`enable_thinking=True` 让推理时模型从 `<think>\n` 开始；训练 completion 没有 `</think>` 关闭段，模型没信号知道什么时候离开 thinking 出答案

详见 `docs/harness_alignment.md` §3。

### 5.2 Prompt BPE ≤ 512 硬 SLA

`max_model_len=4096` − `max_tokens=3584` = 512。超出会在推理时被截断导致 prompt 尾部（guard + role marker）丢失。

- 当前 `src/student/lora_train.py` 的 `_simple_token_count` 只能给空格分词估计
- 必须在 H100 上用真 Nemotron tokenizer 做一次全量 audit
- 超出样本应 **filter，不截断**（截断会破坏 chat template role markers）

### 5.3 推理引擎是 vLLM，不是 transformers + peft

`src/student/inference.py` 用 `transformers.AutoModelForCausalLM + peft.PeftModel.from_pretrained` 只能当本地 dev replica；headline eval 必须加 vLLM 分支（照抄 metric kernel 的 `generate_predictions`）。

`scripts/eval_competition_replica.py` 现状未验证过和 vLLM 语义等价。

### 5.4 Submission 形态只能是单 adapter

harness 只挂第一个 `adapter_config.json`，多 adapter 堆叠无解。
这直接约束了 stage2 / stage3 的交付方式（见 §7）。

---

## 6. 已起草的设计方案

### 6.1 `docs/harness_alignment.md`（PR0 设计说明）

覆盖：
- harness 契约的完整还原（§1 代码片段级别）
- 仓库当前行为（带行号引用）
- 三条 misalignment 的严重度
- 目标最终态：`PROMPT_MODE_CHAT_THINKING`、completion = `{trace_body}\n</think>\n{boxed}`
- 10 处实现落点（3 新 + 7 改 + 4 明确不改）
- 3 条必须在 H100 + 真 tokenizer 下做的实验
- 5 条 PR0 合并验收标准

### 6.2 尚未起草但已规划的文档

| 文件 | 覆盖内容 | 状态 |
| --- | --- | --- |
| `docs/artifact_naming.md` | PR1 + PR2 共享的命名总表（audit_*, smoke_*, canonical） | 未起草 |
| `docs/adapter_delivery.md` | stage2 → stage3 continue training 策略 | 未起草 |
| `scripts/probe_chat_template.py` | §6.1 experiment 脚本（30 行左右） | 未起草 |

---

## 7. 悬而未决的决策（等你拍板）

### 7.1 Adapter 交付路径（最重要）

harness 只支持单 adapter，三条可行路径：

| 方案 | 交付 | 风险 |
| --- | --- | --- |
| **A** 只训 stage2 distill，不做 stage3 repair | 1 个 adapter（stage2） | 放弃 baseline 失败样本的定点修补信号 |
| **B** stage2 + stage3 数据合并，一次性训 | 1 个 adapter（mixed） | 训练信号混合，收敛困难 |
| **C** 先训 stage2 到收敛，再以其 adapter state 为 init 继续训 stage3 数据 | 1 个 adapter（sequential） | 实现稍复杂（需 PeftModel.from_pretrained + continue train） |

作者之前推荐 C，但没 confirm。**这一决定影响 `configs/train_stage3_repair.yaml` 的 `adapter_init_path` 字段是否需要新增、以及 H100 GPU 预算 5h vs 10h**。

### 7.2 Trace 粒度

`docs/harness_alignment.md` §4.2 推荐把现有 `short_trace / token_trace` 的 body 塞进 `<think>` 段，前缀里的 `sig=` 保留 vs. 改成 `sig_bucket=`：

- 保留 `sig=`：每 sample 多 30–80 个 BPE tokens，但保留完整 op 参数信息（solver-level fidelity）
- 改 `sig_bucket=`：更短，只保留 op 序列模板，对 prompt 512 token 预算更友好但损失 solver 信号

因 completion 不是 prompt 的一部分，**`sig=` 的长度不挤压 512 预算**；所以这一决定只影响训练 completion 长度，不影响 prompt SLA。倾向保留 `sig=`。

### 7.3 `artifacts/submission.zip`（1.5 GB）是什么

仓库根目录存在 `submission.zip`，大小 **1,551,980,765 bytes ≈ 1.5 GB**。
LoRA rank 32 的合理 adapter 大小约 100–300 MB。推测 zip 里把 base model 也打进去了 —— 如果这个文件曾被上传过（或将被误上传），harness 会因为找到 Nemotron 权重文件而**行为不可预测**。

**朋友审核的第一件事**：确认这个 zip 是怎么生成的、目前是历史垃圾还是某个脚本的合法产物；
若是垃圾，PR2 里顺手删除，并给 `scripts/validate_submission.py` 加一条 "zip size < 500 MB" 的 preflight assert。

### 7.4 PR 切分和执行顺序

当前规划：

```
PR0 (Prompt/Completion 契约对齐 + audit_prompt_lengths.py)
  ↓
PR0.5 (synth hard family 白名单 + stage2 annotation 旁路)
  ↓
PR1 (默认入口 / README / smoke 脚本收口)
  ‖
PR2 (artifact 命名归一化 + schema 守卫)
  ↓
PR-adapter (stage2 → stage3 continue training 实现，依赖 §7.1 决策)
  ↓
PR-H100 (真实环境验证 + vLLM replica + Nemotron tokenizer 长度 audit)
```

PR1 和 PR2 的 **命名总表**（`docs/artifact_naming.md`）应先于二者合并，否则 PR1 的 smoke 输出名和 PR2 的 canonical 名容易冲突。

---

## 8. 已观察到的异常（审核第一眼要看）

### 8.1 `submission.zip` 1.5 GB（根目录）

见 §7.3。**最高优先级核查项**。

### 8.2 `artifacts/` 有 9 个 adapter_* 目录

```
artifacts/
├── adapter                        # 最早？
├── adapter_demo                   # demo 相关
├── adapter_demo_submission.zip    # 对应打包
├── adapter_official               # 某次全量尝试
├── adapter_official_full_runpod   # Runpod 尝试
├── adapter_smoke                  # smoke 产物
├── adapter_stage1_format          # 当前 canonical
├── adapter_stage2_selected_trace  # 当前 canonical（命名将改成 adapter_stage2_distill）
├── adapter_stage3_repair          # 当前 canonical
├── baseline_eval_smoke.json       # smoke baseline 产物
└── runpod_downloads/              # Runpod 下载缓存
```

其中 **canonical 只有 3 个**（stage1/2/3）。其余 6 个是历史迭代残留，应在 PR2 里迁移到 `artifacts/_archive/` 或直接删。

### 8.3 `data/processed/` 同样新旧并存

见 §4.5 的表格。

### 8.4 `configs/` 有 5 个旧 `train_lora*.yaml`

```
configs/
├── train_lora.yaml                            # src/student/inference.py L132 和 src/student/lora_train.py L475 的默认 config
├── train_lora_official.yaml                   # 未被任何脚本引用（grep 确认）
├── train_lora_official_full_runpod.yaml       # scripts/train_official_full_runpod.sh 在用
├── train_lora_smoke.yaml                      # scripts/train_smoke_local.sh 在用
├── train_lora_stage3_repair.yaml              # 和 train_stage3_repair.yaml 功能并存
├── train_stage1_format.yaml                   # canonical
├── train_stage2_selected_trace.yaml           # canonical（命名将改成 train_stage2_distill）
└── train_stage3_repair.yaml                   # canonical
```

需要在 PR1 里统一：canonical 三个 `train_stage*`；老 `train_lora*` 要么删、要么改名并入，并同步更新 `src/student/inference.py` L132 / `src/student/lora_train.py` L475 的默认值。

### 8.5 `3663529.3663827.pdf`（469 KB）在仓库根

一篇论文 PDF（Chain-of-Event 相关，`codex_takeover_plan.md` 的灵感源）。
不影响功能，但按工程规范应移到 `docs/references/` 或 `.gitignore`。

---

## 9. 测试覆盖快照

`tests/` 下 15 个 pytest 文件：

| 文件 | 覆盖范围 |
| --- | --- |
| `test_answer_extract.py` | boxed 抽取器 |
| `test_atomic_ops.py` | teacher 的 op_catalog |
| `test_chain_search.py` | teacher 的 beam search |
| `test_eval_competition_replica.py` | 本地 replica evaluator |
| `test_format_guard.py` | wrap_boxed |
| `test_lora_train.py` | LoRA 训练的 config / dry-run |
| `test_metrics.py` | competition metrics |
| `test_package_submission.py` | adapter 打包 |
| `test_parser.py` | competition parser |
| `test_program_signature.py` | program signature canonicalization |
| `test_prompt_templates.py` | prompt 模板 |
| `test_research_pipeline.py` | 整链路 smoke |
| `test_sft_dataset_builder.py` | SFT 数据构造 |
| `test_split_builder.py` | 训练/验证拆分 |

最后一次运行：**81 passed**（与 codex_takeover_plan.md 的 milestone 一致）。
**但这些都只覆盖 pure-Python 逻辑**，不覆盖：真实模型加载、真实 tokenizer、vLLM 推理、Kaggle submission harness。

---

## 10. 审核建议路径（按时间预算）

### 10.1 快速 pass（30 分钟）

1. 读本文件（§1–§7）
2. 打开 `docs/harness_alignment.md`，只读 §1 和 §3
3. 直接看 `官方资料/kaggle_metric/nvidia-nemotron-metric.ipynb` 的 `generate_predictions` 函数（cell 0）
4. `Get-ChildItem artifacts/` 和 `Get-ChildItem data/processed/` 看历史垃圾
5. 确认 §7.3 的 1.5 GB submission.zip 情况

### 10.2 中等深度（2 小时）

除 10.1 外，加：

1. `pytest -q` 跑一遍所有单测
2. `bash scripts/train_stage2_distill.sh`（目前 `dry_run: true`）看全链路是否通
3. 读 `src/student/sft_dataset_builder.py`（378 行），理解 stage1/2/3 数据构造差异
4. 读 `src/competition/prompt_templates.py` + `src/teacher/trace_compiler.py`（短）对照 §5.1 的 misalignment
5. 读 `docs/harness_alignment.md` 完整版

### 10.3 完整审核（半天）

除 10.2 外：

1. 按 §4 的六条风险逐条 grep 代码路径验证
2. 按 §8 的异常逐个核查
3. 读 `codex_takeover_plan.md`（717 行，原始设计文档，**部分已过时**；尤其 §11 之前的策略层仍然有效，§13+ 的具体 LoRA 超参和命名已被后续实现覆盖）
4. 用 `kaggle kernels pull` 的几条命令确认权威源最新

### 10.4 如果你准备动手改代码

推荐起点：

1. **先把 §7.3 的 submission.zip 1.5 GB 搞清楚**（可能一分钟就删了）
2. **起草 `docs/artifact_naming.md`**（30 分钟），让 PR1/PR2 都有共享名字表
3. **起草 `scripts/probe_chat_template.py`** 等到 H100 时第一条命令就能跑
4. **先做 PR0**（`docs/harness_alignment.md` 里的实现落点），因为 PR1 依赖它
5. PR0 合并后再做 PR0.5（synth）和 PR1（入口收口）

---

## 11. 关键文件速查表

### 11.1 权威源（只读参考）

| 文件 | 用途 |
| --- | --- |
| `官方资料/kaggle_metric/nvidia-nemotron-metric.ipynb` | **评测本体** |
| `官方资料/kaggle_demo/nvidia-nemotron-submission-demo.ipynb` | LoRA 训练 / 打包示例 |
| `官方资料/competition_data/unzipped/train.csv` | public 训练集 |
| `官方资料/test.csv` | public test |

### 11.2 设计文档

| 文件 | 覆盖 |
| --- | --- |
| `docs/handover.md`（本文件） | 交接全景 |
| `docs/harness_alignment.md` | PR0 设计说明 |
| `codex_takeover_plan.md` | 最早的策略文档（部分过时） |

### 11.3 真实训练入口

| 文件 | 备注 |
| --- | --- |
| `scripts/train_stage1_format_align.sh` | stage1 shell |
| `scripts/train_stage2_distill.sh` | stage2 shell（L24 仍引用旧 config 名） |
| `scripts/train_stage3_repair.sh` | stage3 shell |
| `configs/data_official.yaml` | 官方数据入口（避开 sample fixture） |
| `configs/synth_hard_triads.yaml` | stage2 synth 配置 |
| `configs/train_stage1_format.yaml` | stage1 yaml |
| `configs/train_stage2_selected_trace.yaml` | stage2 yaml（将改名） |
| `configs/train_stage3_repair.yaml` | stage3 yaml |

### 11.4 数据产物（canonical）

| 文件 | 行数 / 大小 |
| --- | --- |
| `data/processed/official_train_tagged.jsonl` | 16 MB |
| `data/processed/stage1_format_align_{train,valid}.jsonl` | 8077 / 1423 |
| `data/processed/stage2_distill_{train,valid}.jsonl` | 4471 / 667 |
| `data/processed/stage3_repair_{train,valid}.jsonl` | 2600 / 476 |
| `data/synthetic/synth_hard_triads.jsonl` | 2000（含 18% easy family，见 §4.3） |
| `data/splits/official/splits.json` | 所有 split 的 id 列表 |

### 11.5 仍需要动的代码热点

| 文件 | 行 | 要改 |
| --- | --- | --- |
| `src/competition/prompt_templates.py` | 全文件 | 加 `PROMPT_MODE_CHAT_THINKING` |
| `src/teacher/trace_compiler.py` | 8–25 | render_* 拆 body + `</think>` wrap |
| `src/student/sft_dataset_builder.py` | 38–46, 200–210, 267–272 | annotation 旁路 / hard family 白名单 / schema 守卫 |
| `src/student/inference.py` | 33, 132 | 对齐 chat template / 默认 config 名 |
| `src/student/lora_train.py` | 78–79, 475 | 真 tokenizer 长度 / 默认 config 名 |
| `scripts/train_smoke_local.sh` | 5, 7, 15, 25 | 旧 synth / split 名全换 |
| `README.md` | 50, 72, 163, 170 | 文档口径对齐 |

---

## 12. 一句话小结（给你的朋友）

"仓库在数据和脚手架层都已经走完 dry-run，但最上游的 **prompt/completion 格式** 没有和官方 evaluator 对齐（拿到 metric kernel 才发现的），这是比任何 config 清理都更紧急的事；与此同时根目录有一个 1.5 GB 的 `submission.zip` 要先确认是什么。读 `docs/harness_alignment.md` 决定是否接受 §4.2 的 'trace-as-thinking' 方案，读本文件 §7 决定三个悬而未决的路径，就可以按 PR0→PR0.5→PR1→PR2→PR-adapter→PR-H100 的顺序推进。"
