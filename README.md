# NVIDIA Nemotron Reasoning Challenge Repo

面向 Kaggle `NVIDIA Nemotron Model Reasoning Challenge` 的 clean-room 研究仓库。

仓库目标分成两条线：

- `teacher`：离线规则图、family tagging、atomic ops、chain search、synthetic data、hard-case mining
- `student`：基于 `NVIDIA Nemotron-3-Nano-30B` 的 LoRA 微调与 `submission.zip` 打包

最终提交物保持为 **LoRA adapter**，不依赖运行时图求解器。

## 当前状态

目前已经完成：

- 官方 `train.csv / test.csv` ingestion
- `\\boxed{}` 提取、本地 exact / numeric evaluator
- family tagging、teacher benchmark、global rule graph、synthetic data、hard-case mining
- Nemotron 官方模型链路 smoke 微调
- Runpod H100 正式全量训练启动

本地单测当前为：

```bash
python -m pytest -q
```

最近一次结果：

- `59 passed`

## 已有产物

本地已下载的官方 smoke 产物：

- `artifacts/runpod_downloads/official_smoke_final/adapter_official_smoke_runpod`
- `artifacts/runpod_downloads/official_smoke_final/submission_official_smoke_runpod.zip`
- `artifacts/runpod_downloads/official_smoke_final/train_official_smoke_runpod.log`

当前根目录下的 `submission.zip` 已指向 smoke 提交包，用于提交探针和格式验证：

- `submission.zip`

## 云端训练

Runpod H100 上正在跑的正式训练配置：

- `configs/train_lora_official_full_runpod.yaml`

相关脚本：

- `scripts/train_official_full_runpod.sh`
- `scripts/watch_and_package_official_full_runpod.sh`

训练完成后，云端会自动生成：

- `artifacts/adapter_official_full_runpod`
- `submission_official_full_runpod.zip`

## 推荐工作流

### 1. 数据准备

```bash
python scripts/prepare_data.py --config configs/data_official.yaml
```

### 2. 离线 teacher / evaluator

```bash
python -m src.experiments.run_baseline --input data/processed/official_train_tagged.jsonl --output data/processed/baseline_eval.json
python -m src.experiments.run_teacher_benchmark --input data/processed/official_train_tagged.jsonl --output data/processed/teacher_benchmark.json --max-per-family 30
python -m src.experiments.run_hardcase_repair --input data/processed/baseline_eval.json --output data/processed/hard_cases.json --max-items 64
python -m src.teacher.synth_generator --config configs/synth.yaml
python -m src.experiments.build_global_rule_graph --input data/processed/teacher_benchmark.json --output data/processed/global_rule_graph.json
```

### 3. 本地 smoke

`5080` 本地适合做小规模 smoke 和数据/teacher 开发，不建议本地硬跑 Nemotron 30B 正式训练。

```bash
python -m src.student.lora_train --config configs/train_lora_smoke.yaml --dry-run
```

### 4. 云端官方训练

```bash
python -m src.student.lora_train --config configs/train_lora_official.yaml
python -m src.student.package_submission --adapter-dir artifacts/adapter_official --output submission.zip
```

对于根盘较小的云端机器，推荐在配置里显式指定缓存目录：

```yaml
environment:
  KAGGLEHUB_CACHE: /workspace/.cache/kagglehub
  HF_HOME: /workspace/.cache/huggingface
  TRANSFORMERS_CACHE: /workspace/.cache/huggingface/transformers
```

## 目录结构

```text
configs/        数据、评测、合成、LoRA 训练配置
data/           原始数据、处理中间件、split、synthetic 占位目录
notebooks/      EDA 与分析 notebook
scripts/        本地与云端工作流脚本
src/common/     IO、路径、日志、seed、文本规范化
src/competition/  parser、schema、metrics、answer extract、split builder
src/teacher/    atomic ops、family tag、graph、search、synth、curriculum
src/student/    SFT dataset、format guard、LoRA train、inference、packaging
src/experiments/ baseline、benchmark、graph build、hard-case repair
tests/          单测与 fixture
```

## 说明

- 仓库默认不跟踪大体积训练产物、缓存和处理后数据
- `artifacts/`、`data/processed/`、`data/synthetic/` 等目录主要用于本地/云端运行时产物
- 如果要提交 Kaggle，最终需要一个根目录下可用的 `submission.zip`

## 下一步

当前最有价值的下一步是：

- 等待云端 full run 完成
- 下载最终 `submission_official_full_runpod.zip`
- 提交 Kaggle 看正式分数
- 视 leaderboard 反馈继续补 teacher / 数据配方
