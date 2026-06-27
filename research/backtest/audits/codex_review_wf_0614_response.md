**结论先行**

这两个裁决的方向大体对，但我不会把它们评为“最终干净”。最大问题是 WF-001 仍疑似有 **r20 标签可得性泄漏**：训练样本按 `trade_date < train_end` 切了，但 r20 标签本身是 `next_open -> close_20d` 的未来收益，靠近训练截止日的样本标签在预测月当时并不可知。也就是说，模型没直接吃预测月特征，但可能吃了预测月/之后的标签信息。

**1. de-lookahead 是否干净**

不完全干净。证据：

- WF-001 月度 r20 训练只用 `trade_date < train_end`：见 [wf001_gen_picks.py](D:/aicoding/stockagent-analysis/research/backtest/wf001_gen_picks.py:73)。
- 但 r20 评估/标签定义是未来 20 日：`next_open` 和 `close_20d = shift(-20)`，见 [walk_forward_validation.py](D:/aicoding/stockagent-analysis/walk_forward_validation.py:147)。
- 因此例如 202410 预测前，若训练含 202409 下旬样本，其 r20 标签要到 202410/202411 才知道。这需要 **20 个交易日左右 embargo/label-availability lag**，当前代码没看到。

所以 WF-001 已经修掉了“生产模型跨全测试期训练”的大头 lookahead，但还不是严格实盘 walk-forward。真实 Sharpe 1.84 可能仍偏高。

smoking gun 推理基本成立，但措辞要收窄：

- 生产 r20 in-sample IC 0.4357，true-OOS 0.1814，WF 全期 0.1007，见 [WF-001.json](D:/aicoding/stockagent-analysis/research/verdicts/WF-001.json:16)。
- 这强烈支持“生产模型 0.44 的高 IC 含记忆/共模 lookahead”。
- 但“+0.44 全是记忆”略过头。更严谨说法是：**0.44 相对 0.18 的超额部分高度疑似记忆；真实 r20 OOS IC 应按约 0.10-0.18 估计，且需 embargo 后再确认。**

另一个小红旗：`r20_oos_diagnostics.csv` 把 202509 标成 `False`，但 `run_wf001.py` 重新按 `<=202509` 算 in-sample。结果 JSON 用后者，数值大方向不变，但这是产物一致性问题。

**2. TP 真改进是否成立**

分两个命题看：

- “具体 +10/+20/+30 档位有择价 alpha”：不成立/证据很弱。
- “均值回归 book 上，赢家分批减仓这种结构能降波动、改善 Sharpe”：中等成立，但需 final holdout。

证据：

- apples-to-apples 的 TP40-base40 ΔSharpe 是 +0.809，baseline_40d Sharpe 1.421，TP_40d Sharpe 2.23，见 [WF-002.json](D:/aicoding/stockagent-analysis/research/verdicts/WF-002.json:8)。
- 年化几乎没变，Δann +0.033pp；maxDD 改善 +14.711pp，见 [WF-002.json](D:/aicoding/stockagent-analysis/research/verdicts/WF-002.json:13)。
- 静态降暴露 ΔSharpe -0.089，说明不是“少持仓就机械抬 Sharpe”。
- 但随机阈值 TP 平均 ΔSharpe +0.769，真实 TP 只多 +0.0397，而且真实 TP 落在随机阈值分布内。这是对“档位技能”的强削弱。

我会把裁定从“TP 真改进”改写为：**结构性分批止盈/赢家减仓在当前 WF picks 样本中显著改善风险调整收益；固定阈值没有证明优于随机阈值，不能声称择价能力。**

可信度：结构性 de-risk 约 6.5/10；固定三档 TP 约 2/10；可 ship 约 4/10，取决于 embargo 后 WF-001 和最终 holdout。

**3. Ship 前必须补的检验**

优先级最高：

1. **WF-001 加 r20 标签 embargo 后重跑。**  
   训练截止不能只按特征日期 `< train_end`，还要保证训练标签在预测月前可知。实操是训练样本 `trade_date <= prediction_start - 20 trading days - 1`，或按 label end date 过滤。WF-002 必须基于这个新版 picks 重跑。

2. **WF-002 重新定义 gate。**  
   不要用 `TP > random mean` 当通过条件。应要求真实 TP 明显高于随机 TP 分布，比如超过 p75/p90，或报告“等价于随机阈值结构”。否则 gate 会把 +0.04 Sharpe 的噪声当胜利。

3. **最终 holdout 只跑一次。**  
   holdout 前冻结：是否采用“任意分批止盈结构”、阈值、成交假设、成本、涨跌停处理、regime 规则。不要再看 holdout 调档。

4. **成交保守性检查。**  
   当前 intraday-high 触发 TP Sharpe 2.23，close-based 只有 1.866，仍好于 baseline 但缩水明显。ship 前要用更保守成交假设做主口径或至少列为 haircut。

5. **随机阈值机制扩展负控。**  
   加“固定持有到首次上涨后卖 1/3”、“随机卖出日期但条件为已盈利”、“按收益分位而非价格阈值”几类，判断 edge 到底是盈利条件、时间缩短，还是阈值触发。

我的最终裁决：WF-001 证明了旧绝对数被注水，但新版绝对数还要经 embargo 重算；WF-002 证明了“赢家分批降风险”值得进入最终 holdout，但没有证明 +10/+20/+30 是有择价 alpha 的最佳止盈策略。
