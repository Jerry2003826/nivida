# Nemotron LoRA 0.54 -> 0.53 复盘

时间：2026-04-24  
对象：`routeweighted mixed router-probe public25 hard30 all30 train15 shared_experts`  
结果：public score = `0.53`，低于当前 best `0.54`

## 一句话结论

这次掉分不是“模型完全没练上”。训练本身确实改变了 LoRA，并且 B thin 曾把 official baseline `0.53` 拉到 `0.54`。真正的问题更像是：我们把 MoE 路由探针结果用错了，尤其是混合 route report 时没有按来源归一化，导致 3 条 visible public test 实际占了约 `51.65%` 的路由权重；再把这些专家 delta 广播进 `shared_experts`，破坏了原来 B thin 的微弱收益。

## 已知分数序列

| 提交 | public score | 说明 |
| --- | ---: | --- |
| Official smoke LoRA baseline | 0.53 | 官方/烟测基线 |
| A-path thin | 0.51 | 明显伤害 |
| stage2_selected_trace | 0.54 | 老 full/selected trace |
| B thin | 0.54 | 训练确实带来恢复/小增益 |
| expertmean-shared top8 scale1 | 0.54 | norm-based expert -> shared，未增益但没掉 |
| routeweighted mixed | 0.53 | route-probe expert -> shared，掉回 baseline |

## 训练是否真的发生了

证据显示训练是有效运行的：

- train records: `9683`
- eval records: `667`
- matched LoRA modules: `93`
- trainable LoRA tensors: `186/186`
- eval loss 下降：`0.5747 -> 0.4023 -> 0.3584 -> 0.3452`
- LoRA B max abs 从初始附近继续增长到约 `0.0136`
- B thin 提交 public score = `0.54`，比 official baseline `0.53` 高

所以“0.54 看起来没练上”的更合理解释是：公榜反馈很粗，且训练目标/代理验证集和隐藏 public metric 不强一致。不是 adapter 没被加载，也不是训练完全无效。

## Routeweighted 为什么会更差

### 1. 混合权重实现有实际偏差

名义设置是：

- public visible: `0.25`
- official_hard: `0.30`
- official_all: `0.30`
- stage2_train: `0.15`

但 `tools/mix_route_reports.py` 是直接对 raw route counts 做 `count * weight` 后相加，没有先按每个来源的 token 总数归一化。

实际每层平均 topk_total：

| 来源 | 行数 | max_new_tokens | 名义权重 | weighted total | 实际贡献 |
| --- | ---: | ---: | ---: | ---: | ---: |
| public_test35 | 3 | 192 | 0.25 | 226344.0 | 51.65% |
| official_hard | 256 | 0 | 0.30 | 78102.0 | 17.82% |
| official_all | 256 | 0 | 0.30 | 61731.0 | 14.09% |
| stage2_train | 512 | 0 | 0.15 | 72087.3 | 16.45% |

也就是说，这次所谓的 mixed route 实际上过度听了 3 条 visible test，而且这 3 条还包含生成阶段 token 的路由，不只是 prompt 路由。visible test 只有 3 行，本来就不该承载这么高的决策权重。

### 2. “常被调用”不等于“该被强化”

路由频率高的专家很可能是通用语法、格式、token 分布专家，而不是决定 bit/encryption/reasoning 答案的专家。把高频专家的 LoRA delta 搬到 shared path，会把原本由 router 条件触发的行为变成所有 token 都吃到的全局偏置。

### 3. shared_experts 不是 routed experts 的等价容器

MoE 里的 routed expert 是条件路径；shared_experts 是全局共享路径。把 routed expert 的 delta tiled/平均后注入 shared_experts，本质是把条件专家的专门行为广播到全局。这个方向天然有风险。

### 4. 路由分布并没有“总是调用某些专家”那么强

混合 route report 一共 23 个路由层：

- top1 平均占比：`7.62%`
- top8 平均占比：`34.74%`
- top16 平均占比：`50.66%`
- 归一化熵均值：`0.86`
- layer 1 / 3 的 top8 只覆盖约 `16.7%`

这说明专家选择还是比较分散的。用 top8 做硬选择，本身就可能选错很多对 hidden public 有用的专家。

### 5. route-based 和 norm-based 选中的专家差异很大

routeweighted 与 norm-based top8 的专家集合重叠很低：

- 每个 module 平均重叠：`1.39 / 8`
- 12 个 module 完全 0 重叠
- routeweighted 的 relative expert norm 更小：`0.178`
- norm-based 的 relative expert norm 更大：`0.231`

norm-based scale1 没涨但维持 `0.54`；routeweighted delta 更小却掉到 `0.53`。这更支持“方向错了”，而不是“扰动太大”。

## 为什么总卡在 0.54

当前证据更像是：

1. B thin 的训练确实带来小收益，但收益只够让 public 从 `0.53` 到 `0.54`。
2. public score 可能粒度很粗，`0.01` 可能只是少数 hidden public 样本差异。
3. 代理 eval_loss 下降不等于最终 exact/hidden metric 上升。
4. visible test 只有 3 行，不能用来代表 public/private 分布。
5. 对 MoE expert 的后处理如果不经过强验证，很容易把小收益抹掉。

## 私榜风险

公榜可以作为最终反馈，但不能作为主要优化目标。私榜更不可见，最稳的策略应该是构建本地任务族验证：

- 按 task family 做 holdout，而不是随机切分。
- 分别看 bit manipulation、encryption、logic、math、format following 等族。
- 指标用最终答案 exact/parsed accuracy，不只看 teacher-forcing eval_loss。
- 任何提交前，至少要确认它没有在多个本地族上系统性伤害 B thin。

## 下一步建议，按省钱优先级

### 不开 GPU也能做

1. 修正 `mix_route_reports.py`：每个 source 每层先归一化成分布，再按 source weight 混合。
2. 重新生成 normalized route report，只比较专家选择变化，不急着提交。
3. 对比四套选择：
   - norm top8
   - raw route top8
   - normalized route top8
   - no-public route top8
4. 看 normalized/no-public 是否仍然与 norm top8 大幅背离；如果大幅背离，route 路线风险仍高。

### 如果再开 GPU，先做短验证，不直接训练

1. 只跑本地推理评估：baseline / B thin / norm-shared / route-shared 在同一批 holdout 上的最终答案准确率。
2. ablation 不要全层注入：
   - scale `0.25/0.5`
   - 只改高 concentration 层，例如 top8 覆盖率高的层
   - 不用 visible public
   - 只 prompt route，不用 generation token route
3. 如果 route 方向仍不稳，就放弃 expert transplant，回到数据与训练目标：
   - answer-only loss
   - task-family balanced sampling
   - harder official-derived examples
   - 多 checkpoint ensemble/selection，而不是继续改 shared_experts

## 我目前会给别人的判断

routeweighted 这次掉分的最大可解释原因是“混合 route report 的统计单位错了”：名义 25% 的 3 条 visible public，因为生成 token 数巨大，实际变成 51.65% 的专家选择来源。再叠加 MoE expert 到 shared_experts 的结构性不等价，导致 B thin 的小收益被抹掉。  

因此下一步不应该立刻开大机器继续冲，而应该先本地修正 route 统计和验证协议；否则继续提交很可能只是围着 `0.53-0.54` 抖。
