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
| T-001 事件研究 (ratio vs MA5 拐点, lead/lag) | done | **sync** (ignition τ0 / peak +2d / trough -7d; choppy ignition 漂移 -5d) |
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
- 迭代1 | T-001 事件研究 | 19月窗口(20240601-20260101) v3c 逐日 causal 推理出 ratio(190万行,落 research/cache/ratio_series.parquet 做 checkpoint); MA5 斜率滞后确认上拐点(零前视, 19万事件/5026股); 60d MA5/MA60 穿越数分 trend/mid/choppy; ±10日相位均值曲线。
  - **裁决 sync**: ratio 不领先拐点。raw 相位曲线 trough≈τ-7 → ignition(最大单日抬升)τ0 → peak τ+2 → 回落。即"同步/确认型"而非"预测型"信号。
  - **关键 regime 发现**: peak 跨 regime 恒为 +2d (不变); 但 ignition 在 choppy 漂移到 -5d (轻度领先), trend/mid 保持 τ0 → 相位关系随 regime 非平稳。**这是 T-002(窗口非平稳)的直接证据/起点。**
  - 教训复盘: 初版用事件内 z-score 提相位形状被重尾污染 + 边界伪峰, 误报 lead@-10d; 改用 raw 均值曲线 + 内区[-8,8]搜地标后干净。中间指标只描述不作 gate (SIGN-R03)。
  - 产出: research/t001_phase_event_study.py / research/cache/ratio_series.parquet / research/cache/t001_results.json / research/cache/figs/t001_phase_curve.png / research/verdicts/T-001.json
  - 下一步 (T-002): 把 ratio 拆短窗/长窗贡献 + 量化 ignition lead/lag 随 regime(快/慢市)的漂移幅度。T-001 已显示 choppy ignition 提前 5d 是漂移存在的强提示 → 预期 status 偏向 window_dynamic_needed, 但须按数据定。ratio_series.parquet 可复用免重算。
