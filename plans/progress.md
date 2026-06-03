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
| RG-001 因果动量-regime 检测器 | todo | — |
| RG-002 v3c regime 分层尸检 (确认假设) | todo | — |
| RG-003 动量-regime 叠加层 | todo | — |
| RG-004 按-regime walk-forward gate | todo | — |
| RG-005 opt-in 落地 (依赖 RG-004=PASS) | skip | — |

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
