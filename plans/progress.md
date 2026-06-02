# Progress — research/ratio-phase

> fresh-context 每轮干净重启, 这里是跨迭代的唯一记忆 (SIGN-R09)。每轮结束前更新。

## 北极星
判定"相位感知 / 动态窗口"对 pump ratio 是否带来经 **walk-forward 验证**、相对生产线 **V12.31**
的真 alpha。有就 opt-in 落地, 没有就把负结论文档化并停手。**生产线 V12.31 全程冻结。**

## 事前注册 gate (冻结, SIGN-R01)
walk-forward 19 月, 同协议同 harness 对照 V12.31:
- Δα(月化) ≥ +0.30pp 且 Sharpe 不降 且 无新增灾难月 且 正α月占比不降 → PASS, 否则 REJECT。

## 任务台账
| id | 状态 | 裁决 |
|----|------|------|
| T-001 事件研究 (ratio vs MA5 拐点, lead/lag) | todo | — |
| T-002 窗口非平稳检验 | todo | — |
| T-003 ratio 轨迹特征 (全因果) | todo | — |
| T-004 多尺度门控启动子 (依赖 T-002) | todo | — |
| T-005 walk-forward 决策 gate | todo | — |
| T-006 opt-in 落地 (依赖 T-005=PASS) | skip | — |

## 关键约束 (摘自 guardrails)
- 负结果=合法完成, 禁止在同段 OOS 反复重调 (R02)
- 中间指标 (IC/precision/lead-lag) ≠ 落地, 只认 walk-forward α (R03)
- 任何 IC/回测/特征前先跑 leakage guard (R04); 见 IC>0.5/Sharpe>10 先查泄漏+ST
- 生产文件 hash 冻结 (R05); ST 源头排除 + 全程分层 (R06)

## 迭代日志
<!-- 每轮 append: 迭代N | task | 做了什么 | 裁决/产出路径 | 下一步 -->
- (init) 脚手架就绪, 等待启动。web-platform prd 已归档为 plans/prd.web-platform.json。
