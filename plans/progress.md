# Progress — research/regime-overlay

> fresh-context 每轮干净重启, 这里是跨迭代的唯一记忆 (SIGN-R09)。每轮结束前更新。

## 北极星
v3c/V12.31 是均值回归(买回调)策略, 实盘审计证实它在**动量延续 regime 被血洗**
(0508→0603: 优秀基金 +11.4% 而 picks -10%, 动量分层 D9-D0 价差 +12.3pp 单调)。
目标: 加**因果动量-regime 叠加层**, 强动量态抑制回调买入/切动量, 经**按-regime-分层的**
walk-forward 验证能改善动量月而不伤反转月才落地; 否则负结论文档化。**生产线 V12.31 冻结。**

## 事前注册 gate (冻结, SIGN-R01 + 必须分 regime SIGN-R11)
19 月 walk-forward 按 regime 分层 vs V12.31:
- 全期净 Δα ≥ +0.30pp 且 **动量月 Δα ≥ +1.0pp** 且 **反转月 Δα ≥ -0.10pp** 且 无新增灾难月 → PASS, 否则 REJECT。

## 任务台账
| id | 状态 | 裁决 |
|----|------|------|
| RG-001 因果动量-regime 检测器 | **done** | built (causal_ok, 202605/202606 印证) |
| RG-002 v3c regime 分层尸检 (确认假设) | **done** | **hypothesis_rejected (反向)** — precondition 失败 |
| RG-003 动量-regime 叠加层 | **skip** | RG-002 precondition 失败触发 |
| RG-004 按-regime walk-forward gate | **skip** | RG-002 precondition 失败触发 |
| RG-005 opt-in 落地 (依赖 RG-004=PASS) | skip | skipped |

## 关键背景 (来自 0603 实盘审计)
- 优秀基金 0508→0603 +7.5~17% (均值 +11.4%); 我们 picks -6~16%; 横截面分位掉到 33-48
- 动量分层: D0(跌最多)前向 -11% → D9(涨最多)+1.2%, 单调, D9-D0 +12.3pp = 强动量行情
- v3c past_r5<0 买回调 = 落 D0-D3 最差桶 (-9.6%)
- 审计脚本 research/audit_live_recs.py (横截面分位指标可复用)

## 关键约束 (摘自 guardrails)
- 负结果=合法完成, 禁止同段 OOS 反复重调 (R02); 中间指标≠落地只认 walk-forward α (R03)
- 任何 IC/回测/特征前先跑 leakage guard (R04); 生产文件 hash 冻结 (R05); ST 源头排除 (R06)
- **R11: 评估必须按 regime 分层, 禁止只看全期平均** (这是漏掉实盘错配的根因)

## 迭代日志
<!-- 每轮 append: 迭代N | task | 做了什么 | 裁决/产出路径 | 下一步 -->
- (init) 由 ratio-phase 循环(已完结, REJECT)转入。实盘审计发现 v3c 动量 regime 错配 → 本目标。等待启动。
- 迭代2 | RG-002 v3c regime 分层尸检 | research/rg002_regime_autopsy.py。两路证据。**Part B 主证据=纯因果动量分位尸检** (805 日, regime 来自 RG-001 causal, fwd20 仅作尸检测量非特征 → 无 lookahead): 逐日 ST 排除后按过去20d动量分10档, 测低动量档(D0-D3=v3c回调猎场)前向20d alpha。**结果反转假设**: 条件于因果动量 regime, v3c 回调 α=**+0.378pp 显著为正** (t=4.06 p=0.0001 pos日62%), 反转月 +0.250pp。机理: 因果动量态平均**跟随均值回归** (前向 D9-D0=-1.77pp<0), 回调买入反被奖励。审计"动量月血洗"**未能泛化**: 全样本仅 202303(D9-D0+8.2)/202602(+3.3) 两个极端动量月真血洗=尾部 episode, 非系统 regime 效应。Part A (复用 t005 V12.31-等价 walk-forward 月度 α 按主导 regime 分层): 动量月 α=+3.88pp / 反转月 +4.17pp, 动量月亦无血洗 (但绝对 α 含共模 r20 池 lookahead, 仅旁证)。**诚实限制**: ①实盘血洗窗 0508-0603 需 6 月+前向数据(未到), 完全在尸检窗外(数据止 0506)测不到该 episode; ②2026 动量日 v3c α 已转弱负(-0.094pp), 个别强动量月(202303/202602)确有血洗。裁决 **status=hypothesis_rejected**。据 RG-002 事前注册 precondition (α 非 regime 依赖→叠加层无意义), **RG-003/004 置 skip, RG-005 skipped**。产出 research/cache/rg002_daily_decile.parquet + rg002_results.json + verdicts/RG-002.json。**关键教训**: 一个 compelling 的 n=1 实盘血洗轶事未通过 frozen-regime 全样本因果尸检 — 因果可测的动量 regime 并不预测前向动量延续(反而预示均值回归), 在该 regime 上 gate 回调买入会**伤而非帮**。反过拟合脚手架第二次实战否决 (承 ratio-phase REJECT)。**下一步**: 全部任务有裁决, 循环可完成。若 2026-07 后 6 月前向数据到位, 可开新循环用正确注册的 gate 复检 2026 弱负信号(但须重新事前注册, 禁在本 frozen-regime 上重调)。
- 迭代1 | RG-001 因果动量-regime 检测器 | research/rg001_regime_detector.py。核心信号=**因果动量持续性价差**: t 日按过去 [t-40,t-20] 的 20d 动量横截面分 10 档, 量 [t-20,t] 已实现收益的 D9-D0 价差 (两窗连续全过去, 零前视); +平滑(5d)+宽度(breadth_ma20)+等权指数动量(index_mom20)。固定 deadband ±1.0pp 分 {momentum/mixed/reversal}。**因果性自检 abs_diff=0.0** (抹未来价格重算 spread 完全一致)。ST 源头排除(266 只)。产出 research/features/regime_timeline.parquet (825 日 × 7 列, 20230101-20260603)。分布: momentum 29%/mixed 16%/reversal 55%。**关键吻合**: 已知动量月 202605 判 momentum 83% spread +8.87pp; 202606 (审计 0508→0603 血洗窗) momentum 100% +13.39pp — 因果检测器独立印证实盘审计的强动量行情。裁决 status=built。**下一步 RG-002**: 把 V12.31 的 19 月 walk-forward 按本 regime 分层尸检, 确认"动量月 α 显著负 / 反转月 α 正"假设 (若 hypothesis_rejected 则 RG-003/004 置 skip)。可复用 regime_timeline.parquet 按 trade_date merge。
