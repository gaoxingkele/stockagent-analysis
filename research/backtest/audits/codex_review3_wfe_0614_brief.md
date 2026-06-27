你已两次审计这个 A 股均值回归选股系统 V12.31。你第二次 review 抓到两点: ①WF-001 缺 label embargo; ②WF-002 gate 太松(用随机均值非分布上分位)。我们照你的 5 条清单做了 embargo 干净收尾。请做**第三次 review**: 判断 embargo 是否到位、止盈"结构也不成立"裁定是否成立、Sharpe~1.31 能否当 V12.31 最终真实基线、还有无残留盲点、下一步。只分析不改文件。可只读 research/verdicts/WFE-001.json, WFE-002.json, research/backtest/。

## WFE-001: label embargo r20 真walk-forward 重跑
- embargo: 训练截止收紧到 P_start - 21 交易日 (保证训练样本前向20日 r20 label 在预测前全部可知)。
- embargo 后真·真实: 年化 +34.9% 净Sharpe +1.31 maxDD -15.7% 月胜率70%。
- 缩水链: BT-002 注水版 2.78 → WF-001 de-lookahead 1.84 → WFE-001 +embargo 1.31 (embargo 再削 ΔSharpe -0.53)。仍跑赢 hs300(0.46)/CSI1000(0.81)/动量(0.51)。
- ★疑点: embargo 后 WF r20 全期 IC +0.0979 vs 无embargo +0.1007 (几乎没变, -0.0028), 但 book Sharpe 却掉 0.53。IC 几乎不变而 Sharpe 掉这么多, 你觉得正常吗? 会不会是少数 cohort / 月份构成变化 / 样本太短(20 cohort)导致的不稳, 而非真实信息损失?

## WFE-002: 强化版止盈复测 (embargo picks + close-based + 随机p90 + 4控分解)
- 主口径 close-based 保守成交: baseline_40d Sharpe +1.26 → TP_40d_close +1.38, ΔSharpe=+0.117 (WF-002 报的+0.809 = 日内high乐观成交+0.279 + lookahead picks; 收紧后塌到+0.117)。
- 静态降暴露 placebo -0.008≈0 (非少持仓抬升)。
- 强化 gate: 随机阈值TP(50seed,close-based) p90=+0.279; TP +0.117 < p90 → 择价alpha=False。
- block bootstrap 95%CI=[-0.242,+0.523] 含0; leave-one-out 符号不稳; regime mixed+0.15/mom-0.45/rev-0.19。
- 4控分解 edge 主源=收益分位档位(+0.116)但全在噪声内。
- 裁定=**结构也不成立**: close-based下结构性减仓 CI 含0 ≤ placebo → 弃止盈, WF-002 的 +0.809 是乐观成交+算术假象。

## 我(claude)的解读
1. V12.31 真实可交易 Sharpe≈1.31(注水2.78/de-look1.84/embargo1.31), 仍跑赢全基准 = 真实但绝对预期是naive回测的~一半, 可交易。
2. 止盈在最干净口径(embargo+close-based+随机p90强gate)下死透, 用户止盈直觉被否, +0.809是成交乐观假象。脚手架+你的两次catch 把一个看似+1.36的"胜利"逐步证伪到0。

## 请 review 四点
1. embargo 现在到位了吗? 还有无残留泄漏?
2. ★IC几乎不变(-0.0028)但Sharpe掉0.53 —— 这个不一致你怎么看? 是否说明 1.31 本身也被20-cohort短样本的噪声主导, 置信区间很宽?
3. "止盈结构也不成立"成立吗? 这个干净否定可信吗?
4. Sharpe~1.31 能否定为 V12.31 最终真实基线? 下一步: (a)triple-barrier重标 (b)直接拿1.31去部署/最终holdout (c)别的? 给优先级。
