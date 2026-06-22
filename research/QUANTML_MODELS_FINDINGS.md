# QuantML-Qlib 方法过 gate 发现 (2026-06-19)

> 用户分享 QuantML-Qlib 48篇合集 (SOTA模型/因子挖掘算法/Alpha-Beta解耦)。我们的洞察: 23个被否
> 假设全是 **feature 侧**, 从未测过 **模型 class / 目标函数 / 自动搜索** 是否漏信号。这是唯一未测轴。
> 用户要求 Tier1+2+3 顺序全做。每个: 同 factor_lab 特征 → 换模型/目标/搜索 → walk-forward gate + 多重检验。
> 生产线 V12.31 全程冻结。

## 裁决: 四轴全 REJECT — 模型/目标/解耦/搜索 均非杠杆

| 实验 | 测什么 | 结果 |
|---|---|---|
| **QML-1** 排序目标函数 | r20 regression vs lambdarank/rank_xendcg | ❌ REJECT_cheap: 回归 RankIC +0.1145 vs 0.074/0.080 (NW-t −3.66/−2.68), 排序显著劣 |
| **QML-2** NN 模型 class | torch MLP vs LGBM 同236特征 | ❌ REJECT_cheap: LGBM +0.0953 vs MLP +0.0660 (NW-t −1.87), GBDT 仍最优 |
| **QML-3** Alpha/Beta 解耦 | beta-中性选股 vs 原始 (ABCM/FactorVAE) | ❌ REJECT: book Δα −0.381pp/月, 去 beta 反伤 (edge 与 beta 耦合) |
| **QML-4** 自动因子挖掘 | 随机符号搜 80 价量因子 | ❌ REJECT_known_families: 24 存活全是已 walk-forward 否决族换皮 |

## 各实验要点

### QML-1 排序目标函数 (文章33)
文章主张 MSE 错配排序任务。但我们的 r20 已用 `objective=regression + spearman_ic 早停`——早停已优化排序。换 lambdarank/rank_xendcg (r20 分桶) 反而**丢信息**, 19月 OOS RankIC 显著更低 (NW-t −3.66)。**假设证伪。**

### QML-2 NN 模型 class (LNN/iTransformer/FactorVAE 等)
torch MLP vs LGBM 同 236 工程特征, 19月 walk-forward: MLP 更差更噪 (202501 崩 −0.11)。**GBDT 对 A 股截面 tabular 是极强基线, NN 非线性不漏信号。** 序列-NN-on-raw-OHLCV 未测 (CPU 昂贵 + 236 特征已编码时序, 先验低)。

### QML-3 Alpha/Beta 解耦 (ABCM 文章46 / FactorVAE 文章44)
把 pred_r20 对 beta_60 横截面中性化再选股, book walk-forward Δα = **−0.381pp/月** (14/19月负, 动量/反转都伤)。**去 beta 反伤**——印证市场状态发现 (0618): 我们 book 的 alpha bull-loaded、与 beta 耦合。market-neutral 的 alpha/beta 解耦**不适用于 long-only 净多头 book**(牛市要那个 beta)。

### QML-4 自动因子挖掘 (GP/RL-AlphaGen/LLM)
随机符号搜索 80 个价量因子 → 24 个过正交闸 (扣 FULL_CTRL) + Bonferroni。**但 24 存活全是已 walk-forward 否决的因子族换皮**: 最强 `tsstd_intraday_60` = OA-VOL 的 `intraday_vol_60` (OA-VOL-GATE 已 REJECT Δα −0.15); intraday/overnight 族 = OA-VP (已 REJECT Δα −0.70); max_ret = 彩票 (已否); sum_ret_60 = 动量 (控制)。**自动搜索在枯竭价量空间精确重新发现我们已证 walk-forward 失败的因子族 = AI Scientist 教训 (gate>吞吐) 的完美实证。**

## 元结论

**QuantML-Qlib 48篇方法对我们净增量 = 0。** 四个轴 (目标/NN/解耦/搜索) 全 REJECT, **关掉了"是不是模型不行"——不是模型不行, 是数据/信号枯竭。**

- 文中 PoC IC (0.023-0.087) 正是我们 LGBM 样本内 +0.44→真 OOS 0.18、RETRO +54% 反转、Hybrid +54% stratification artifact 的同类——过不了 walk-forward + 多重检验。
- 与 23 个 feature 侧被否 + 正交 campaign (0618) + 市场状态 (0618) **完全一致收敛**: 瓶颈是价量空间信号枯竭 + 样本功率。
- **增量不在选股** (模型/特征/搜索都试尽), 在 **book/风控/部署** 或 **真正新数据** (微结构/盘口/分析师修正, 当前 Tushare token 无权限)。

## 唯一未测 (CPU 昂贵, 先验低)
序列-NN-on-raw-OHLCV (PatchTST/Mamba/LNN on 原始序列)。但 236 工程特征已编码时序信息, 且 23+4 否定一致指向数据非模型 → 不投。

生产线 V12.31 全程只读冻结。reviewer/gate 全程: NW-t / Bonferroni / placebo / 廉价闸先杀。
