# 市场状态暴露 overlay 发现 (Track A, 2026-06-18)

> 启发: paper *In Good and in Bad Times? The Relation between Anomaly Returns and Market States* —
> 横截面因子 alpha 熊市更强 (41% 非对称, 75% 来自空头端, 套利限制驱动)。
> 问题: 用事前市场状态 (GOOD_TIMES = MA200 + 3年趋势, 沪深300+中证500) 动态缩放 V12.31 book
> 暴露能否改善风险调整收益? 生产线 V12.31 全程冻结。

## 裁决: REJECT —— paper 对 long-only 净多头 book **反指**

## 核心发现

### 1. 我们的 book 在牛市表现最好 (绝对 Sharpe 单调上升)
| GOOD_TIMES | 月数 | 绝对 Sharpe | 年化 |
|---|---|---|---|
| 过渡 (0.5) | 8 | +0.81 | +15.7% |
| 偏牛 (0.75) | 1 | +1.26 | +28.3% |
| **牛 (1.0)** | 11 | **+1.65** | **+54.7%** |

### 2. 超额 alpha (真 alpha) 也 bull-loaded —— 与 paper 双维度全反转
| GOOD_TIMES | 超额 hs300 | 超额 csi1000 |
|---|---|---|
| 过渡 (0.5) | +0.35% | +0.16% |
| **牛 (1.0)** | **+2.82%** | **+2.06%** |

不只绝对收益靠 beta——**选股 alpha 本身就集中在牛市**。regime 维 (momentum +1.96% / reversal +1.78%) 对称, 说明是 GOOD_TIMES (牛熊 level) 维度 bull-loaded, 非动量/反转维度。

### 3. overlay 5 臂测试 (de-lookahead WFE-001 book, 净 Sharpe 1.31)
| 臂 | Sharpe | 年化 | maxDD | 最差月 |
|---|---|---|---|---|
| base (静态全暴露) | 1.306 | 34.8% | -15.7% | -10.4% |
| **dynamic_paper (熊加牛减)** | **1.207** | 16.2% | -10.1% | -4.2% |
| trend (牛加熊减) | 1.319 | 31.9% | -15.7% | -10.4% |
| static_dele (常数均暴露) | 1.306 | 18.7% | -8.9% | -5.7% |

- **paper 规则反指**: 在我们最强的牛市砍仓 → Sharpe 1.31→1.21, 收益腰斩。
- **placebo p=0.588**: dynamic 与随机择时无异 → **零 timing 技能**。
- **static_dele Sharpe ≡ base**: 证 dynamic 的 maxDD 改善纯是"降暴露"不是 timing 技能。

## 机制 (为什么 paper 不适用于我们)
1. **吃不到空头端**: paper 的熊市 alpha 75% 来自做空高估股; A 股做空约束极致 = paper 机制最强, 但我们 long-only **不可收割**。
2. **启动子系统牛市最有效**: V12.31 的 pump 启动子 + 动量捕捉在牛市离散度最大时 alpha 最高。
3. **净多头 beta 顺风**: 牛市 beta 加持。
三者叠加 → 我们的 edge 集中在牛市, 与市场中性因子的熊市 alpha 相反。

## 局限 (诚实)
de-lookahead 诚实 book 仅 2024-10~2026-06 (walk-forward 需训练史), **无熊月** → paper 的"熊市加仓"侧无诚实数据可测 (不补 lookahead 避免误导数)。但绝对+超额 alpha 均单调随 GOOD_TIMES 上升 + 机制清晰 → 反转结论 prior 强。

## 可落地结论
- **不实施市场状态暴露择时** (对我们反指或无 timing 技能)。
- V12.31 应保持**充分投资** (edge 在牛市)。
- 若未来要管熊市尾部风险, 应走 **beta 趋势保护** (index < MA200 减仓) 而非 paper 的 alpha 择时——但需真熊市数据 OOS 验证, 当前无。

## 元结论
paper "alpha 熊市强"是 **long-short 市场中性 + 空头端套利限制**现象, **不可移植到 long-only A 股净多头 book**。这是数据诚实告诉我们的边界, 非 paper 错。反过拟合脚手架 (placebo 负控 + static_dele 隔离降暴露 + 事前 GOOD_TIMES) 全到位, 真小未夸大为 PASS。
