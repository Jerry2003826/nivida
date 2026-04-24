# Routeweighted 0.53 复盘 v2：采纳审核后的收紧版

时间：2026-04-24  
目标提交：`routeweighted mixed router-probe public25 hard30 all30 train15 shared_experts`  
结果：public score = `0.53`，低于当前 best `0.54`

## 最收紧结论

这次最强的可解释问题，不是“routeweighted 选错了几个 expert”，而是两层问题叠加：

1. route signal 的统计口径和名义权重不一致：配置看起来表达 source-level mixture weight，但实现实际执行 token-count-weighted mixture，让 token 多的来源天然支配选择，尤其是 3 条 visible public + generation token。
2. 后处理目标不成立：把 routed expert delta 注入 `shared_experts`，等于把条件专家路径广播成全局 bias。

所以 routeweighted 掉到 `0.53`，更像是方向性错误，而不是扰动过大、训练没生效、或者提交没加载。

## 现有证据边界

当前 probe JSON 是 source aggregate，不含 per-example route counts。因此：

- 可以做 source-level normalized / no-public / leave-one-out 分析。
- 不能从现有产物恢复 per-example normalized route。
- 不能把 public run 拆成 prompt-only 与 generation-only，因为 `public_test35.json` 的 prompt/generation counts 已经混在一起。

source-level normalized 只能修正 source 之间的 token 总量不平衡，不能修正同一 source 内部长 prompt / 长 generation 样本支配短样本的问题。最终理想口径仍然是：

```text
per-example per-layer normalized
-> source average
-> source weighted mixture
```

per-example normalized 需要重跑 route probe。当前不应马上开 GPU 做训练或新提交；如果后续开 GPU，优先重跑 probe / inference eval，而不是训练。

## 已补充的 aggregate stability 结果

分析脚本：

`tools/analyze_route_selection_stability.py`

输出：

`data/processed/route_probe/selection_stability_aggregate.json`

### 1. raw count 混合的统计口径偏离

名义权重是：

```text
public_visible 0.25
official_hard  0.30
official_all   0.30
stage2_train   0.15
```

但因为混合脚本直接使用 raw route count，visible public 在 mixed route counts 中的 per-layer average raw-count share 变成：

| 来源 | 行数 | max_new_tokens | 名义权重 | 实际 raw-count share |
| --- | ---: | ---: | ---: | ---: |
| public_visible | 3 | 192 | 0.25 | 0.5165 |
| official_hard | 256 | 0 | 0.30 | 0.1782 |
| official_all | 256 | 0 | 0.30 | 0.1409 |
| stage2_train | 512 | 0 | 0.15 | 0.1645 |

这说明 visible public 在 route-count 统计里被严重放大。这里的 `0.5165` 是 raw route count 混合后的计数贡献，不是最终模型行为影响力的精确比例；放大的也不是 3 条 prompt 本身，而是 prompt + generation token 的混合轨迹。

### 2. raw mixed vs source-normalized mixed

| 比较 | mean top8 overlap | min | max | mean Jaccard | mean JSD |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_mixed vs source_normalized_mixed | 6.83 / 8 | 4 | 8 | 0.758 | 0.0055 |

解读：

raw-count mixing 与名义 source 权重不一致的问题确实存在，但修成 source-level normalized 后，top8 并不会完全翻盘。也就是说，不能把掉分全部归因于“public overweight 选错 top8”。更稳的解释是：统计口径不一致是一个硬问题，但 shared transplant 的方法论风险也必须同时承担责任。

### 3. raw mixed vs no-public source-normalized

| 比较 | mean top8 overlap | min | max | mean Jaccard | mean JSD |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_mixed vs no_public_source_normalized | 6.04 / 8 | 3 | 7 | 0.623 | 0.0212 |
| source_normalized_mixed vs no_public_source_normalized | 7.13 / 8 | 6 | 8 | 0.814 | 0.0054 |

解读：

去掉 public 后，raw mixed 的选择变化明显大于 source-normalized mixed。这支持审核意见：visible public 不应该参与 selection，最多做 diagnostic。

### 4. public visible 与 official 分布的差异

| 比较 | mean top8 overlap | min | max | mean Jaccard | mean JSD |
| --- | ---: | ---: | ---: | ---: | ---: |
| public_visible vs official_all | 3.48 / 8 | 1 | 6 | 0.297 | 0.1301 |
| public_visible vs official_hard | 5.65 / 8 | 2 | 7 | 0.564 | 0.0485 |
| official_hard vs official_all | 4.35 / 8 | 2 | 6 | 0.385 | 0.0732 |

解读：

public_visible 和 official_all 差异很大。visible public 只有 3 条，不能代表 public/private；它和 hard case 更接近一些，但这个接近可能只是题型偶然重叠，不能作为 selection 权重依据。

### 5. source leverage

定义：

```text
leverage(source, layer) =
    8 - |top8(full_mix, layer) ∩ top8(remove_source, layer)|
```

raw mixed 下：

| 删除来源 | mean leverage | max leverage |
| --- | ---: | ---: |
| public_visible | 1.83 | 5 |
| official_hard | 0.35 | 2 |
| official_all | 0.48 | 2 |
| stage2_train | 0.26 | 2 |

source-normalized mixed 下：

| 删除来源 | mean leverage | max leverage |
| --- | ---: | ---: |
| public_visible | 0.87 | 2 |
| official_hard | 0.83 | 2 |
| official_all | 1.00 | 3 |
| stage2_train | 0.17 | 1 |

解读：

raw mixed 时 public_visible 是最高 leverage source，说明 3 条 visible public 在原实现里确实对 selection 有不成比例的影响。归一化后 public leverage 回落，official_all 反而更重要，这才比较符合我们想表达的 source 权重。

## 方法论风险：route frequency 不是 expert utility

route count 只能说明：

```text
某 expert 经常被 router 选中
```

但我们真正需要的是：

```text
某 expert 的 LoRA delta 对最终答案正确性有正向边际贡献
```

高频 expert 可能负责的是：

```text
标点、换行、模板、常见英文 token、数字表面分布、格式习惯、通用语法
```

这些 expert 被注入 `shared_experts` 后，可能不会增强推理，反而污染所有 token 的 hidden state。尤其 exact-answer benchmark 对格式、边界条件和最终 token 非常敏感，错误方向的小扰动就足够把 `0.54` 抹回 `0.53`。

## shared_experts transplant 应降级为高风险路径

结构上：

```text
routed expert  = conditional, sparse, token-specific
shared expert  = unconditional, dense, every-token
```

所以 transplant 不是“把有用 expert 共享化”，而是：

```text
把条件行为变成全局 bias
```

已观测结果也支持保守判断：

```text
B thin                       0.54
norm-based shared top8 s1     0.54
routeweighted shared          0.53
```

norm-based 没证明有增益，但至少没明显破坏；route-based 更像有害方向。即使修好 route normalization，也不能直接推出 route-shared 会变好。

## 下一步实验顺序

### 第一步：不开 GPU 训练，继续做 selection stability

当前可做：

```text
S_norm_top8
S_raw_route_top8
S_source_norm_route_top8
S_no_public_route_top8
```

需要重跑 probe 才能做，且重跑应优先服务于诊断，不直接训练或提交：

```text
S_per_example_norm_route_top8
S_prompt_only_route_top8
S_generation_only_route_top8
```

重跑前必须改探针，让每个 example、每层都有独立 route distribution，否则还是会被长样本支配。

### 第二步：如果再开 GPU，先 inference，不训练

至少比较：

```text
official baseline
B thin
norm-shared scale1
raw-route-shared
source-normalized-route-shared
no-public-route-shared
```

指标必须是：

```text
parsed exact accuracy
final answer exact match
format-valid rate
task-family accuracy
```

不要只看 loss。按 family 拆开看 bit manipulation、encryption、logic、math、format following、long-context/multi-step。

### 第三步：如果仍要做 shared variant，只做低风险交集

不要再每层固定 top8。优先：

```text
S_safe = S_norm_top8 ∩ S_no_public_normalized_route_top8
```

再加过滤：

```text
scale = 0.125 / 0.25
只改 top8_mass 高的层
只改 entropy 低的层
不用 visible public
prompt-only route
norm 太小不动
public leverage 太高不动
```

如果每层交集只有 1-3 个 expert，就只动 1-3 个，不要为了凑 top8 注入低置信方向。

## 更有前途的主线

当前最稳主线仍然是保留 `B thin`，转向训练目标和验证协议：

```text
answer-only loss
task-family balanced sampling
hard negative / official-derived augmentation
format-constrained decoding validation
checkpoint selection by parsed exact accuracy
```

这比继续围绕 public leaderboard 调 shared_experts 更可能突破 `0.54`。

## 给外部 reviewer 的最终表述

这次 routeweighted 失败有两个层级的问题：

第一层是统计口径不一致：配置里的 source weight 看起来表示 source-level 权重，但实现实际执行 token-count-weighted mixture，导致 token 数更多的 source 被额外放大。3 条 visible public 在 mixed route counts 中的 per-layer average raw-count share 从名义 25% 变成了约 51.65%。  

第二层是方法论风险：route frequency 不是 expert utility，把 routed expert delta 注入 `shared_experts` 会破坏 MoE 的条件性，把 sparse conditional behavior 变成 dense global bias。  

所以不建议马上继续开 GPU 冲 route-shared。先做 per-example normalized、no-public、prompt-only route report，再看它和 norm-based selection 是否有稳定交集。没有稳定交集的话，route transplant 应该降级，不再作为主优化方向。
