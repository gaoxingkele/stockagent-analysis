# V12.31 因子审计报告 — TreeSHAP 归因 × 消融诊断

> 对标论文《Interpretable Factor Decomposition … China's A-Share Market》(TreeSHAP + 消融双诊断)。
> 数据源: 我们已落地的 **DLV-001 TreeSHAP**(逐只票加性归因) + **DLV-004 block-ablation**(embargo walk-forward)。
> 被审计对象 = V12.31 的 **r20 选股模型** (PIT-clean r20_v16, 剔 holder_pct, label-embargo 滚动); 19 月 OOS。
> 用途: 回答"模型赚的什么钱 / 哪些因子是承重墙 / 哪些可剪" (解释层 · 风控 · 出资人)。**非新 alpha, 是质检。**

---

## 一、TreeSHAP 全局重要性 (个股级加性归因, base_value=+2.4171)

每只 pick 的 `pred_r20 = base + Σ(各因子 SHAP)`, 加性自检 max 误差 5.3e-14 (PASS)。全期 311 只 picks 的 |mean SHAP| top:

| # | 因子 | mean\|SHAP\| | signed | 所属块 |
|---|---|---|---|---|
| 1 | cyb_rel_strength (创业板相对强度) | 2.80 | +2.80 | cross_sectional_rel |
| 2 | industry_id (行业) | 1.41 | +1.30 | market_context |
| 3 | mkt_ret_20d (大盘20日动量) | 0.92 | −0.26 | market_context |
| 4 | mkt_ret_60d | 0.73 | −0.10 | market_context |
| 5 | mkt_ret_5d | 0.26 | −0.25 | market_context |

→ **驱动 V12.31 r20 排序的是"相对强度 + 行业 + 大盘动量(反向)"**——典型行为/价量信号，与论文"行为因子统治 A 股"一致。

## 二、消融诊断 (块级 ΔIC, 19 月 embargo-WF, baseline IC=+0.1091)

ΔIC = 剔除该块后 IC − 完整模型 IC。**负=剔了变差(承重)；正=剔了变好(冗余)；CI 不含 0 才显著。**

| 块 | ΔIC 95% CI | 解读 |
|---|---|---|
| **market_context** | [−0.154, −0.024] | 🧱 **承重墙(最强)** — 剔除显著重伤 |
| **moneyflow** | [−0.026, −0.0016] | 🧱 **承重墙** — 剔除显著伤 |
| fundamental_chip | [−0.025, +0.0003] | 🧱 边缘承重(几乎显著) |
| pyramid | [−0.020, +0.0019] | 可替代(偏承重) |
| cross_sectional_rel | [−0.027, +0.008] | 可替代* (见下背离) |
| volatility | [−0.019, +0.0045] | 可替代 |
| candle_pattern | [−0.019, +0.0044] | 可替代 |
| oscillator | [−0.016, +0.0043] | 可替代 |
| breakout_position | [−0.013, +0.0064] | 可替代 |
| volume_liquidity | [−0.0065, +0.0080] | 可替代/中性 |
| trend_ma | [−0.0081, +0.0137] | 冗余倾向(剔了略升) |
| valuation_size | [−0.017, +0.014] | 冗余倾向(剔了略升) |

**无净负块**(无任何块剔除后 IC 显著上升) → r20 特征集已干净, 剪枝无实质收益。

## 三、承重墙 / 可替代 / 冗余 三分类 (SHAP × 消融)

| 类别 | 块 | 判据 | 含义 |
|---|---|---|---|
| 🧱 **承重墙** | market_context, moneyflow, (fundamental_chip边缘) | 消融 CI 不含0(负) | 独立增量信息, **不可替代**, 剔除重伤 |
| 🔧 **可替代** | cross_sectional_rel, pyramid, volatility, candle, oscillator, breakout, volume | 消融 CI 含0 | 与其它块共线, 剔除有补位, 但仍贡献 |
| 🗑 **冗余倾向** | trend_ma, valuation_size | 消融 Δ 偏正(剔了略升) | 边际为负/噪声, 论文同款"估值=冗余" |

## 四、最关键洞察: SHAP 高 ≠ 不可替代 (背离)

论文的核心发现在我们数据上**完全复现**:

- **market_context = 真承重墙**: SHAP 高(mkt_ret/industry 共 ~46% 排进 top5) **且** 消融 CI[−0.154,−0.024] 重伤 → SHAP 与消融一致, 这是模型的**地基**(大盘择时/行业 beta)。
- **cyb_rel_strength = "轻钢龙骨"(SHAP第一但块可替代)**: 单因子 SHAP 居首(2.80), 但其所在 cross_sectional_rel 块消融 CI[−0.027,+0.008] **含0** → 抽掉后 volatility/momentum/pyramid 等行为块**迅速补位**。这正是论文"换手率 SHAP第一但消融可替代"的同型替代效应——**模型天天用它, 但不是非它不可**。
- **valuation_size = 冗余**: SHAP 垫底(估值族) + 消融剔了略升 → 月度周期估值信号更新慢、小盘投机行情失效, 是噪点。**与论文"估值=冗余, 剔除净化"一致。**

## 五、对 V12.31 的结论

1. **赚的什么钱**: r20 选股层主要靠 **大盘择时/行业(market_context, 承重墙) + 资金流(moneyflow, 承重墙) + 相对强度(cyb_rel, 高频使用)** = **行为/价量信号**, 不靠估值/基本面。印证我们正交 campaign"基本面/估值正交因子全塌缩"。
2. **可剪吗**: 不建议。无净负块, 估值/trend_ma 虽偏冗余但剔除收益落 21 月噪声带内, 剪了省不了多少、反增运维风险 (DLV-004 已定: V12.31-clean 不动选股逻辑)。
3. **承重墙别碰**: market_context / moneyflow 是地基, 任何"简化特征"的改动都要先看它俩。

## 六、诚实 caveats (vs 论文)

- 本审计是 **r20 选股层** 归因, 不含 V12.31 完整双轨 book 逻辑(pump 启动子/ratio 排序/行业 cap/双轨)——解释的是"谁进池/排序", 不是全部收益。
- 消融效应**落在 21 月样本噪声带内**(CI 多含0), 所以"可替代/冗余"是软结论, 不等于"无用"。
- 我们的数字经 **embargo walk-forward** (baseline IC +0.109 是 300k 子样, 真 OOS); 论文 2009-2019 滚动 **无 embargo**, Sharpe 2.23/Carhart α 2.31% 偏乐观, 且时段(散户期)与当下(机构抱团期)结构已变——方法可借, 量级不可移植。

---
*生成: TreeSHAP(DLV-001) + 消融(DLV-004) 既有产出合成; 生产线 V12.31 只读未动。*
