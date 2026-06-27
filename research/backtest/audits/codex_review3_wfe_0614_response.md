**结论**

1. **Embargo 基本到位，置信度中高。**  
   证据在 [wfe001_gen_picks.py](D:/aicoding/stockagent-analysis/research/backtest/wfe001_gen_picks.py:59)：`EMBARGO_TDAYS = R20_HORIZON + 1 = 21`；[wfe001_gen_picks.py](D:/aicoding/stockagent-analysis/research/backtest/wfe001_gen_picks.py:99) 用 `trade_date <= embargo_cut` 训练；[wfe001_gen_picks.py](D:/aicoding/stockagent-analysis/research/backtest/wfe001_gen_picks.py:71) 从预测月首个交易日往前切 21 个交易日。这个设计能覆盖 r20 label availability。  
   残留盲点不是明显泄漏，而是未在本次限定文件内完全证明：`compute_r20_label()` 的精确定义、复权/财务特征是否含未来修正、以及缓存模型是否确实由新版 embargo 代码重建。

2. **IC 几乎不变但 Sharpe 掉 0.53，不矛盾。**  
   IC 是全截面、等权、诊断性 rank correlation；book Sharpe 是经过选股阈值、行业/双轨构造、仓位、成本、再平衡、尾部路径后的组合结果。小 IC 变化可以对应较大的组合 Sharpe 变化，尤其当被 embargo 切掉的是“靠近训练截止、对边际入选股票最有影响”的样本。  
   但你的怀疑成立：20 cohort 太短，`1.31` 的置信区间会很宽。WFE-001 没有像 WFE-002 那样给 baseline 做 cohort bootstrap/LOO，所以不能把 `1.31` 当高精度参数，只能当当前最干净点估计。证据：WFE-001 verdict 记录 Sharpe `1.306`、无 embargo `1.838`、IC `0.0979`，见 [WFE-001.json](D:/aicoding/stockagent-analysis/research/verdicts/WFE-001.json:7) 和 [WFE-001.json](D:/aicoding/stockagent-analysis/research/verdicts/WFE-001.json:16)。

3. **“止盈结构也不成立”成立，置信度高。**  
   WFE-002 的 gate 是干净的：主口径 close-based，`TP_40d_close - baseline_40d`，随机阈值用 p90，不是均值；bootstrap CI 含 0；LOO 符号不稳。证据在 [run_wfe002.py](D:/aicoding/stockagent-analysis/research/backtest/run_wfe002.py:355)、[run_wfe002.py](D:/aicoding/stockagent-analysis/research/backtest/run_wfe002.py:388)、[run_wfe002.py](D:/aicoding/stockagent-analysis/research/backtest/run_wfe002.py:473)。结果字段也一致：ΔSharpe `+0.1168`，random p90 `+0.2795`，`struct_improve=false`，见 [WFE-002.json](D:/aicoding/stockagent-analysis/research/verdicts/WFE-002.json:12)、[WFE-002.json](D:/aicoding/stockagent-analysis/research/verdicts/WFE-002.json:19)、[WFE-002.json](D:/aicoding/stockagent-analysis/research/verdicts/WFE-002.json:21)。  
   我的裁定：弃固定档位止盈。它不是“还需要调参”，而是当前证据下没有资格进入生产假设。

4. **Sharpe ~1.31 可以定为 V12.31 当前最干净真实基线，但不能定为最终可部署期望。**  
   更准确表述：`V12.31 clean WFE baseline point estimate = net Sharpe ~1.31`。它替代 `1.84/2.78`，但必须带不确定性标签。

**下一步优先级**

1. **先冻结 V12.31 @ 1.31，做最终 untouched holdout / forward paper trading。** 这是验证“能不能部署”的唯一高优先级动作。  
2. **给 WFE-001 baseline 补 cohort bootstrap、LOO、按月份/市场状态归因。** 不是改策略，是给 `1.31` 加误差条。  
3. **停止推进当前止盈模块。** WFE-002 已经足够否定。  
4. **triple-barrier 重标作为 V12.32 研究分支，而不是替代 V12.31 基线。** 它可能改善标签结构，但属于新模型假设，必须重新走 embargo WFE + holdout，不能拿来“修饰”这次基线。
