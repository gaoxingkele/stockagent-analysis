# SmartGoal Debate: 提高入选池配对盈利率（Dream-RSI 探索）

Date: 2026-09-20

## Context

用户问题：入选池最终赚钱比例（A∪B）作为单一目标如何继续提高；可否用 RSI 制定探索计划。这里的 RSI 是 Recursive Self-Improvement（Dream-RSI / Harness-RSI），不是已否决的 Wilder 相对强弱。

当前发运 π0（非正式 H04，不进生产）：

- 冻结 multinomial_base Top20，惩罚效用 `pA−1·pB−2·pD−0.25·pC`
- 独立 P(B10) 滚动校准后卡门，当前门关闭（τ=1）
- 主档 +15% 可提前走 / −10% 路径风险
- 外层 320 只：盈利 **56.3%**，配对 −10% **5.9%**，A 179 / B 1 / C 122 / D 18
- 回放 τ=0.26 盈利 59.4%→60.6%、回撤持平 3.3%，外层与 π0 相同，未迁移

已失败或冻结的路线：

- Wilder RSI 特征增量（打乱对照不差于真 RSI）
- 按配对四类重训排序（外层 55.3%/11.9%，未超过冻结排序）
- 未校准 P(B10) 当绝对门槛；外层挑 τ
- 旧 Dream-RSI Top50+μ=3+max_risk=0.6（B5 持有口径，未迁移）
- 生产 R20 / pool_e / 网页 / 21:00

决策目标：在 **不放松配对 −10% 约束、不向外层搜参** 的前提下，用 Dream-RSI 回放世界提高入选池 A∪B。

## Expert Positions

### Expert A: Time-Series

**提案：** 先把任务说清楚——这不是再训一个 20 日收益预测器，而是 **决策层**：在冻结四类概率轨迹上，改排序定义、弃权与日历门控。C 类是「占资不赚」，对排序损失应直接惩罚 `pC` 或改用 `pA+pB` / 校准后的 P(U)。不要上 TCN/Transformer，直到选择层残差证明分数本身不可分 A/C。

**理由：** 外层失败结构是 122 个 C 对 18 个 D。路径风险已经低；再卡门是在切 C 的同时切 A。τ=0.26 能换 2 只 A←C，说明边际还在 **名单替换**，不在模型容量。

**实验：** 冻结分数上有限政策网格（ν、hold_profit、min_score），再镜像 B10 做 **延迟标签 rolling Platt on logit(pA+pB) 或 pC**，用校准后的盈利分排序。

**成功：** 回放段 A∪B 严格高于 π0，配对 −10% 不差于 π0，相对 π0 的覆盖（20×日）≥0.3，外层再次严格提高盈利且风险不差。

**风险：** 校准后的 P(U) 会像未校准 P(B10) 一样被当成绝对概率；A/C 基率在回放/外层漂移。

### Expert B: Quant Factors

**提案：** 单一目标可以是 **最大化入选盈利率，约束为路径 −10% ≤ π0**。N 是上限不是配额，允许空位，但禁止靠「每天只留 3 只明星」虚增胜率。优先打 C：提高 ν、按 `pA+pB` 排、对低 `pA` 弃权。T+1、提前止盈已在标签里，不要改 +15/−10 配对去贴盈利。

**理由：** 买入推荐池的人关心「买进去有没有赚」，也关心会不会先砸 −10%。把风险从效用里拿掉、只留独立门，排序才能专门对付 C。流动性/涨跌停暂不作为本轮增量（诊断池已过滤）。

**实验：** 同日同 k 比较选股；全日历报告空选。覆盖分母固定为 π0 的 20×信号日，不用挑战者自己的 n_cap。

**成功：** 外层盈利差 >0，且 −10% 差 ≤0；最差月披露；空选胜率为 null。

**风险：** 小 N、高 ν 在回放段好看，外层把高 ν 变成少选低波动 coiling 股，C 率更高。

### Expert C: Mathematical Theory

**提案：** 信息瓶颈在 **A vs C**，不在 B∪D。B10 实现率外层全体 20.6%、入选 5.9%，风险头对入选子集几乎饱和；盈利头（U=A∪B）入选 56.3% vs 需要分开测量的全体基率。先做入选子集上 pA、pA+pB、pC 的可靠性（延迟标签校准），再谈换模型。Dream-RSI 的「世界」应保持：同一冻结轨迹、同一延迟时钟，只换政策。Cunningham：回放改善 ≠ 自加速，外层必须独立否证。

**理由：** τ=0.26 回放换 C→A、外层零差异，说明该门槛对应的校准质量在两段上的 **分位数位置不同**，不是一个稳定的 A/C 分割。

**实验：** 回放段构造 τ/min_score 只用分数质量（分位数），不用标签；标签只用于评价。泄漏检验：`label_available_at < prediction_at`。

**成功：** 校准后 P(U) 在外层入选子集上 mean vs realized 的差距小于原始分；若差距不缩，停止把校准分当排序。

**风险：** 用回放标签调 ν 再在同一回放上宣称改善，是一次信息消耗；必须预先登记网格。

### Expert D: Benchmark Integration

**提案：** 开一轮 **Harness-RSI L1**，脚本独立于生产。π0 必须在网格里。发运规则改为：目标=盈利；约束=配对 −10% ≤ π0 且相对 π0 覆盖 ≥0.3；外层同样约束下盈利严格更高才替换。不改冻结排序权重，除非 L1 证明分数不可分且残差实验先过。Wiki 记下否决项。

**理由：** 现有 `policy_replay.default_grid` 是 B5 口径的 N/μ/max_risk，从未在配对 +15/−10 + 校准 P(B10) 上重跑。这是最低成本的可证伪下一步。覆盖若按挑战者自己的 n_cap 计算，缩小 N 会自动「满覆盖」。

**实验：** 新诊断入口 `scripts/replay_s20_v4_profit_rsi.py`，产物 `output/experiments/s20_safe_v4/sources/v4-paired-campaign/profit_rsi.json`，测试覆盖分母与 π0 保留。

**成功：** 有发运记录：推荐政策、Dream vs outer 表、`formal_training_authorized=false`。无迁移则 π0 留任仍算该轮完成。

**风险：** 把「完成 S20 最优化」写成生产升级；H01 仍是 FAILED_VALIDITY。

## Disagreements

| Issue | Positions | Resolution experiment |
|---|---|---|
| 能否丢掉 −10% 只冲盈利率 | A/B/C：否，作约束；若当无约束目标会选回高波动 | 任何挑战者若回放 −10% > π0，直接无资格，不看盈利 |
| 缩小 N 是否算提高胜率 | B/D：N 是上限，但覆盖分母钉死 20×日；A：允许 Top10 若约束满足 | 预先登记 n_cap∈{10,20,50}；覆盖=`n_selected/(20×dates)` |
| 要不要立刻重训排序 | A/C：L1 饱和且 A/C 校准失败后再说；已有配对重训负结果 | L1 无合格挑战者且校准 P(U) 在入选子集仍系统性偏离，才开残差备忘，不开训练 |
| 日级空选 vs 只数弃权 | B：先个股 min_score；A：C 若按日成团再做日门 | 先 E-P3；按日 C 率聚类显著再登记日门，仍只在回放段定阈值 |

## Prioritized Goals

| Rank | Goal | Why now | Acceptance criteria | Artifact |
|---:|---|---|---|---|
| 1 | L1 政策网格打 C：ν / hold_profit / pA / min_score | 失败主体是 C，卡门已经试过 | 回放盈利>π0 且 −10%≤π0 且覆盖≥0.3×π0；外层同样才发运 | `profit_rsi.json` |
| 2 | 延迟校准 P(U) 或 pC 后再排序 | B10 校准已证明分数≠概率 | 外层入选子集校准均值靠近实现；排序挑战者须走同一发运规则 | `p_u_calibrated.parquet` |
| 3 | 覆盖与空选会计钉死相对 π0 | 防止小 N 虚增 | 测试：n_cap=10 填满时 coverage 相对 20 槽为 0.5 而非 1.0 | `test_policy_replay.py` |
| 4 | 日级门控（仅当 E-P3 显示 C 按日成团） | 可能是择时不是选股 | 全日历空选胜率=null；外层盈利约束内提高 | 后置 |
| 5 | 特征/模型残差（研究-only） | 负结果已多 | 无 L1/校准失败证据不开 | 备忘，不训练 |

## Experiment Plan

| Experiment | Data | Method | Metrics | Failure criteria |
|---|---|---|---|---|
| E-P1 C 惩罚 | 冻结 cal_p_A..D + 配对 labels_tp15_dd10 + p_b10_cal | ν∈{0.25,0.5,1.0,2.0}，π0 在内，回放选、外层确认 | A∪B、B∪D、相对 π0 覆盖、ABCD | 回放不双约束改善，或外层不迁移 → 留 π0 |
| E-P2 排名定义 | 同上 | ranking∈{penalized_utility, hold_profit, safe_probability}；n_cap∈{10,20,50} | 同上；次档 +25/−15 只报 | 仅 n_cap=10 提高盈利率但相对覆盖<0.3 → 无资格 |
| E-P3 弃权 | 同上 | min_score 分位数只来自回放段分数质量（pA 或 pA+pB），不用标签、不用外层 | 同上 | 空账或覆盖不足不当改善 |
| E-P4 滚动 P(U) | samples 的 prediction_at / label_available_at | expanding Platt on logit(pA+pB) 或 pC，成熟标签 strict before 预测日 | 入选/全体 mean vs realized；再用校准分走 E-P2 | 校准不改偏差或泄漏测试失败 → 停 |
| E-P5 日成团诊断 | E-P1/3 的入选名单 | 按日 C 率、是否集中少数信号日 | 日间方差 vs 置换 | 无成团 → 不做日门 |

失败分类沿用：NO_GAIN / TRADEOFF / INSUFFICIENT_EVIDENCE。不改 +15/−10，不向外层加 τ，不启用 Wilder RSI，不重训 multinomial 当本轮升级。

## ARA / Wiki Updates

- 本文件为本轮专家记忆。
- 发运对照：`docs/research/s20_v4_opt_shipped_policy_20260920.md`
- 下一份实验记录应写 `docs/research/s20_v4_profit_rsi_explore_YYYYMMDD.md`，并写明 formal_training_authorized=false。
- 代码入口建议：`research/s20_harness/policy_replay.py`（min_score、相对 π0 覆盖、n_cap=10）、`scripts/replay_s20_v4_profit_rsi.py`、`tests/s20_harness/test_policy_replay.py`。
- 只读冻结排序：`output/experiments/s20_safe_v4/sources/v4-rsi-campaign`；写入 `v4-paired-campaign`。

## Decision

用 Dream-RSI 把 **历史推荐轨迹当世界**，只进化选择政策（Harness-RSI L1）。单一目标 = 入选池配对盈利率；配对 −10% 与相对 π0 覆盖是硬约束，不是第二效用项。先打 C 类（ν / P(U) 排序 / 弃权 / 延迟校准），不再收紧 B10 门。π0 始终在候选集；外层不迁移则继续发运 π0。这不是正式 H04，不是生产 S20。

## Addendum: 沉默股票（最高涨幅 <8% 且未破 −10%）

用户提议把周期内最高价相对入场 <8%、且未破 −10% 的票剔除进沉默池。诊断（T+1 后 20 根 high/low，与配对标签同一窗口）：

- 全体 28.2% 沉默（15,685/55,529）。其中 C 9,689、A 5,996；**没有 B/D**。C 的 81% 是沉默，A 的 21% 也是沉默（小涨赢家）。
- 外层 Top20：320 里 **208 只（65%）沉默**（C 112 + A 96）。回放 Top20：180 里 100 只（56%）沉默（C 59 + A 41）。
- 事后删掉沉默（偷看未来）：外层剩 112 只，盈利 56.3%→75%，−10% 5.9%→17.0%。补位后盈利 76.9%，−10% 15.3%。回放同样：盈利升到 79%，回撤 3.3%→11.7%。**盈利升、风险升，双改善不成立。**
- 实体持续性：下次信号仍沉默 | 本次沉默 = 66.5%；本次不沉默则只有 13.1%。

含义：当前排序本来就爱低波动；沉默里既有死钱 C 也有 3%～7% 的 A。预测日不能用尚未发生的最高价。可做的因果版本只有：① 成熟后实体冷却（`label_available_at` 之后进沉默池）；② 独立预测 P(沉默) 当硬门，回放登记、外层确认，且 −10% 不得差于 π0。8% 由用户提出，不在外层改。
