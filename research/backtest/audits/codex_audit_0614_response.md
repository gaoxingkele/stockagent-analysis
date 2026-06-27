使用 `$analyze` 只读核实了 `research/verdicts/*.json`、`research/backtest/`、`plans/prd.json`。未修改文件；`git status` 只显示既有脏文件和 `.omx/` 未跟踪项。

**1. 方法盲点**

证据支持：你们的反过拟合脚手架很强，尤其是预注册、负结果合法、只认 walk-forward、ST 源头排除、regime 分层、消融和共模 Δ 设计。`BT-003` 明确把 r20 池标为 book α 核心，去掉后 `ΔSharpe -1.373`；`pump_down` 完全中性；绝对 Sharpe 仍标注非可交易。`EX-001`/`EX-002` 也都明确写了“描述性/受控 Δ，非 ship gate”。

主要盲点不是 p-hack，而是**问题定义锁死**：

- **label 锁死**：大量否定是在 `r20` 或 pump 启动子标签框架内完成的。若 alpha 存在于更短/更长 horizon、路径质量、尾部规避、容量约束或相对收益，而不是固定 `r20`/启动子命中率，现有 gate 会系统性低估。
- **universe 锁死**：V7c 池和 `past_r5<0` 均值回归入口本身很强地筛掉趋势跟随、质量大盘、机构拥挤等机会。19 连否更像“在这个池和目标函数里，新价格/形态特征没有边际”，不是“市场没有别的信号”。
- **评测口径锁死**：多数结论用同一 book 引擎、同一 r20 共模、同一持仓/行业 cap/双轨结构。受控 Δ 比绝对值可信，但也可能只是在这个 book 结构下成立。
- **样本长度偏短**：核心 book 结论多为 19-21 个月，regime 分层后月数更少，`mixed` 只有 2 个月。对 Sharpe、maxDD、单月 outlier 的置信度有限。
- **共模 lookahead 会污染“候选池成分”**：即使 Δ 相消，lookahead 选出来的是更“理想的路径样本”。任何出场/止盈/风控结论都可能在真实 r20 walk-forward picks 上缩小。

所以，19 连否不说明方法坏；反而说明你们没有轻易把中间指标 ship。但它确实提示：现有研究机器可能过度擅长否定“同类特征增量”，不擅长发现“换目标函数 / 换 universe / 换风险约束”的增量。

**2. EX-001 的 TP-only `+1.36 Sharpe` 是否可信**

我的判断：**方向中等可信，量级不应信。**

可信的部分：

- `EX-001` 代码确实是同一批 V12.31 picks、同一入场，只换出场规则。r20 选股 lookahead 对 TP-vs-baseline 的一阶差分会大幅相消。
- close-based 保守口径下，`TP_only` Sharpe 仍从 baseline `2.40` 上方提升到 `3.1859`，说明结果不是纯靠日内 high 触发的乐观成交。
- TP-only 同时降低年化 `-13.31pp`、改善 maxDD `+5.64pp`，这更像“牺牲右尾换低波动”，符合均值回归持仓的经济直觉。

不可信或需折价的部分：

- baseline 是固定 20d，TP arm 是 40d backstop，比较混入了**出场规则 + 时间暴露 + 空仓现金路径**，不是纯止盈阈值效果。
- 样本只有 20 个 cohort，Sharpe 差对路径和少数 cohort 很敏感。
- r20 lookahead 选出的样本可能更容易先触发 +10/+20/+30%，真实 walk-forward picks 的触发分布可能不同。
- daily bar 无法完全解决同日路径、流动性、盘口排队；TP-only 比 TP+SL 少一些同日顺序问题，但仍有成交质量风险。

干净验证方式：

1. 先生成 de-lookahead r20 walk-forward picks。
2. 固定同一批真实 WF picks，跑 baseline 20d、baseline 40d、TP-only 40d、TP-only close-based 四臂。
3. 报 per-cohort Δ、block bootstrap CI、单月/单 cohort leave-one-out、regime 分层。
4. 加 placebo exit：随机 TP 阈值、反向“早卖输家”、固定降波动现金化对照，确认不是单纯降低暴露带来的 Sharpe 数学提升。
5. 只在未用于设计 TP 网格的新时间段做最终裁决。

**3. 下一步优先级**

1. **(a) 先 de-lookahead 拿 r20 真实 walk-forward P&L。**  
   这是最高价值、也是所有后续判断的地基。`BT-003` 已显示 r20 池贡献几乎全部 book α；只要 r20 真实 WF 不成立，EX-001/002/003 的落地意义都会重估。

2. **(d) 止盈做正式 walk-forward 验证。**  
   TP-only 是当前最便宜、最有经济直觉、且已有受控 Δ 支持的风险改进候选。但必须基于真实 WF picks，且补 baseline 40d 和 placebo，避免把“少持仓/少波动”误判为 alpha。

3. **(b) EX-003 r20 triple-barrier 重标。**  
   价值高，但自由度也高。应在 (a) 建好真实 baseline、(d) 确认 TP 不是 artifact 后做。否则容易把出场发现反向灌进 label，形成循环自证。

4. **(c) 风格 tilt 往大盘质量靠。**  
   EX-002 的风格差异证据很强，但它更像风险管理/产品定位问题，不一定是 alpha 增强。若过早 tilt，可能直接稀释 r20 均值回归小盘成长 edge。应先作为 overlay 小网格或风险预算约束验证。

5. **(e) 我会加一个“审计 holdout/负控包”。**  
   固定一个后续不可再调时间窗；所有 TP、triple-barrier、style tilt 只允许一次进入。并加入 per-cohort bootstrap、同 horizon 对照、placebo exit、容量/停牌/涨跌停成交敏感性。这样能专门防当前最大风险：受控 Δ 看似干净，但真实可交易样本一换就塌。
