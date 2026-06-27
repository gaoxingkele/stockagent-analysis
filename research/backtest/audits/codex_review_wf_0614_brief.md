你之前(0614)审计过这个 A 股均值回归选股系统 V12.31 并给了三条修正。我们按你的修正做了两个实验, 现在请你**review 这两个裁决**: 判断方法是否干净、结论是否成立、有没有残留盲点、下一步建议。只做分析, 不改文件。可只读核实 research/verdicts/WF-001.json, WF-002.json, research/backtest/。

## 背景
你上次指出: ①所有绝对 Sharpe(+140%/2.78)含 r20 共模 lookahead 注水; ②EX-001 止盈"+1.36 Sharpe"混了 baseline-20d vs TP-40d 持有期 + 降暴露, 须 de-lookahead + baseline-40d + placebo 负控。我们照做了。

## WF-001: de-lookahead r20 真实 walk-forward
- r20 月度重训(24m lookback, 19月, time-split val), 每预测月只用之前数据 → OOS 分数 → 重建 V12.31 book 过同一引擎。
- 真实可交易: 年化 +65.9% 净Sharpe +1.84 maxDD -18.0% 月胜率65% (vs 注水版 +140.5%/2.78, 缩水 ΔSharpe -0.94)。仍跑赢 hs300(Sharpe 0.46)/CSI1000(0.81)。
- smoking gun: 生产 r20 IC in-sample(≤202509,19月里12月在其训练窗内) +0.436 → true-OOS +0.181 (衰减0.25); WF 月度重训在同 true-OOS 段 IC +0.184 ≈ 生产 → 生产 +0.44 全是记忆, 真实 r20 OOS IC ≈ +0.10~0.18。

## WF-002: 修正版止盈复测 (在 WF-001 真实 OOS picks 上)
- baseline_40d: Sharpe +1.42; TP_40d(三档+10/+20/+30%各1/3): Sharpe +2.23。
- apples-to-apples ΔSharpe(TP40−base40) = +0.809 (年化几乎不变, maxDD改善14.7pp)。对比 EX-001 混淆口径(TP40−base20) +0.556 的差就是持有期错配+lookahead。
- placebo-A 静态降暴露(f=0.677匹配TP平均暴露) ΔSharpe -0.089 ≈0 → 静态现金混合不抬Sharpe, TP增益来自时变路径。
- placebo-B 随机阈值TP(30 seed) ΔSharpe +0.769 [p5,p95]=(+0.521,+1.085) → 随机分批减仓也得+0.77, TP +0.809 仅略胜。
- block bootstrap(1000×) ΔSharpe 95%CI=[+0.146,+1.328] 不含0; leave-one-out [+0.380,+1.088] 符号稳定; regime: mixed -0.43/momentum +0.76/reversal +0.16pp。
- gate 4条件全过 → 裁定 "TP真改进"。

## 我(claude)的解读
1. V12.31 真实可交易但绝对预期要腰斩(Sharpe~1.84非2.78), r20 真OOS IC~0.1-0.18。
2. 止盈是真改进(+0.81干净, 过静态降暴露placebo决定性), 但**具体档位不重要**(随机阈值≈TP), edge 是"在均值回归book上遇强分批减仓"这个结构, 不是择价技能。

## 请 review 三点
1. de-lookahead 做干净了吗? smoking gun(生产IC 0.44→0.18=记忆)推理成立吗? 还有没有残留 lookahead/偷看?
2. "TP真改进"成立吗? **随机阈值placebo +0.769 ≈ TP +0.809** —— 这是削弱(说明只是平凡的降暴露择时)还是只是说明"档位不重要但结构真"? 这个裁定该信几分?
3. 在把止盈放进最终 holdout / 考虑 ship 之前, 还有什么必须补的检验或盲点? 下一步优先级?
