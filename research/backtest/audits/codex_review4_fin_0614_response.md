**结论**

这三项裁决总体成立，但措辞要更保守：V12.31 可以作为“当前最干净的冻结候选基线”进入前向 paper-trade；不能把 `Sharpe 1.31` 作为已可承诺的真实优势，更不能直接上真钱部署或加杠杆。FIN-003 对“预注册这组 triple-barrier 挑战者”是干净 REJECT，但不能逻辑外推为“所有 triple-barrier idea 永久无效”。

我只读了 [FIN-001.json](D:/aicoding/stockagent-analysis/research/verdicts/FIN-001.json)、[FIN-002.json](D:/aicoding/stockagent-analysis/research/verdicts/FIN-002.json)、[FIN-003.json](D:/aicoding/stockagent-analysis/research/verdicts/FIN-003.json)、[V12.31_BASELINE.md](D:/aicoding/stockagent-analysis/research/backtest/V12.31_BASELINE.md)，未改文件。

**1. FIN-001 CI 解读**

成立，但建议把“1.31真实”改成“1.31是当前最可信点估计”。

`95% CI=[0.11,2.60]` 的正下沿、`P(Sharpe>0)=99%`、LOO cohort `1.02~1.53`、删最重月仍 `1.05`，支持“不是单月/单cohort撑起来的伪优势”。所以第三轮后“从注水 2.78 剥到 1.31”的真实化工作是有效的。

但 CI 太宽意味着：优势方向较可信，优势幅度不可信。`P(Sharpe>1)=66%` 只能说“中等概率超过 1”，不能说“真实 Sharpe 就是 1.31”。尤其样本只有 21 个月，bootstrap 的概率本身也是条件在当前样本结构、当前策略选择路径上的估计，不应过度金融化解读。

我的裁决：FIN-001 通过，但官方话术应是：

> V12.31 的干净 OOS 点估计为 Sharpe 1.31；证据支持正 alpha，但样本功率不足，1.31 不是可承诺下限，前向验证前只能作为 paper-trade 期望锚点。

**2. FIN-003 triple-barrier REJECT**

对“这一个预注册挑战者”REJECT 很干净。

证据链够强：净 Sharpe `0.72 < 1.31`，ΔSharpe `-0.59`；maxDD 更差；bootstrap `P(Δ>0)=10%`；删掉 TB 最有利月后更差；momentum/reversal 两个主要 regime 都伤；五个 gate 全 False。这个不是“略输但有潜力”，而是核心业务指标全面不如基线。

但你们问得对：这不能证明 triple-barrier idea 本身不行。它证明的是：

> 在预注册参数 `+15%/-8%/40d`、更严格 embargo、TB-score 替代 r20 filter、其余口径与 V12.31 对齐的条件下，TB 挑战者不应替换 V12.31。

如果未来要重开，只能作为新策略族，必须重新预注册参数搜索协议、独立验证窗、惩罚多重试验；不能拿 FIN-003 之后继续搜 `+10/-6/30`、`+20/-10/60`，再回来声称“同一个挑战者修好了”。当前收尾阶段，不建议重开。

**3. holder_pct PIT 风险**

真钱上线前必须修；paper-trade 可以带着红标并行观察，但不能视为生产可接受风险。

原因很简单：你们这轮工作的核心价值就是去前视和去注水。`holder_pct` 如果按 `end_date` 而不是 `ann_date` 合并，属于典型季度披露前视，性质上比“单字段影响可能有限”更严重。即使只有 1/235 个特征，它在模型、树分裂、交互项和池过滤里可能有非线性放大；不能只用“单因子不主导”来免罪。

上线前最低要求：核实 factor_lab 的 as-of 口径，或直接替换成已建的 `ann_date` PIT 面板。若只做 paper-trade，允许先跑，但报告里要把它列为“阻止真钱部署”的 open risk。

**4. 最终行动建议**

建议收尾动作如下：

1. 冻结 V12.31，不再调参，不吸收 FIN-003，不复活止盈。
2. 启动 V12.31@1.31 前向 paper-trade，append-only 记录 picks 和 realized P&L。
3. 在真钱部署前修掉 `holder_pct` PIT，确认模型重建严格使用 embargo walk-forward，不准 in-sample 单模型。
4. 至少等 `>=6` 个完整 cohort，且跨过你们标注的 2026-07 压力窗口后，重跑 FIN-001 同口径误差条。
5. 若前向 Sharpe 落在 CI 下半区或 `<1.0`，降级为研究策略；若维持在上半区，再讨论小资金、低容量、无杠杆试运行。

可能仍漏掉的系统性盲点：

- **多重试验幸存者偏差**：20 个实验都被诚实判了，但 V12.31 本身仍是长期研究路径的幸存者，CI 没完全惩罚策略发现过程。
- **真实执行摩擦**：小盘流动性、涨跌停排队、开盘冲击、停牌、成交量容量，比 close-based 成本模型更脏。
- **数据供应商 PIT/修订风险**：不止 holder_pct，日频字段也要防 vendor 后修、复权、成分状态回填。
- **退市/不可交易样本处理**：若 universe 或行情源对退市、长期停牌、ST 转换有 survivorship 残留，会继续抬高回测。
- **regime 外推**：21 个月覆盖有限，A 股风格切换、监管、微盘拥挤、量化踩踏都可能让均值回归边际失效。
- **生产一致性**：研究脚本、每日推理、模型缓存、交易日历、ST 过滤、价格口径任何一个不一致，都会让 1.31 失去对应关系。

最终裁决：**部署 V12.31@1.31 进入冻结前向 paper-trade 是正确收尾；不建议真钱生产部署，直到 holder_pct PIT 修复且前向复检通过。**
