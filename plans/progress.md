# Progress — research/seat-sleeve

> fresh-context 每轮干净重启, 跨迭代唯一记忆 (SIGN-R09)。每轮结束前更新。

## 北极星
把已证真正交的龙虎榜席位印记 (SEAT-002: 残差IC+0.0547 t8.58 全regime, 扛过动量+市场消融;
但接 V12.31 r20池 walk-forward REJECT=载体错) 做成**正确载体 = 独立短线(r5)事件 sleeve**:
仅龙虎榜上榜股 D+1 建仓、短持有(H=5)、跟高战绩席位 top-N。**成败核心 = 真实成本 + 容量 + 净α**。
**关键对照: 跟"高战绩席位"(Arm_A) vs 跟"任意龙虎榜买入"(Arm_B/M)** 隔离席位技能 vs 龙虎榜效应。
**生产线 V12.31 冻结, 本 sleeve 独立 book。**

## 事前注册 gate (冻结 R01 + 分regime R11 + 消融 R12++ + 真实成本)
- 成本: round-trip ~30bps(佣金+印花税0.05%卖+滑点), 敏感 0/15/30。
- gate: net月化α(Arm_A−等权市场)>=+0.50pp |t|>=3 (≥30月事件对齐) 且 Arm_A−Arm_B>0显著(席位技能净增量) 且 Sharpe>=1 且 单月outlier剔后净α>0 且 分regime不集中 且 扣[动量+市场]存活。

## 应对手册 (冻结)
- PASS→独立sleeve落地日频输出; 真小(净α>0但<0.5或Sharpe<1)→记promising待优化不强上;
- REJECT_成本吃掉(毛α>0但net≤0)→文档化信号真但不经济; REJECT_龙虎榜效应(Arm_A≈Arm_B)→不是席位技能; REJECT_负→干净否。

## 任务台账
| id | 任务 | 状态 | 裁决 |
|----|------|------|------|
| SL-001 | r5席位sleeve回测引擎(D+1+席位排序+成本+三臂对照) | todo | — |
| SL-002 | 成本×持有期×容量敏感性 | todo | — |
| SL-003 | walk-forward gate(net-of-cost α+席位技能净增量) | todo | — |
| SL-004 | 独立sleeve日频输出(仅PASS) | skip | — |

## 已有资产
- research/features/seat_footprint.parquet (SEAT-001: sf_edge_r5/r20/winrate_r5, 因果防泄漏, 9235席位/2722可信)
- output/tushare_cache/top_inst.parquet (龙虎榜席位); daily cache (D+1开盘建仓/平仓 + 前向); RG-001 regime; verify.py(isGate)
- 上轮教训 [[project_seat_footprint_real_signal_wrong_vehicle_0606]]: 信号真(+0.05 r5)但r20池载体错; 这轮用r5正确载体

## 关键约束
- 负结果=合法完成禁同段OOS重调(R02); 中间指标≠落地只认net-of-cost walk-forward α(R03); 泄漏闸(R04, 席位战绩只用过去/D+1生效)
- 生产hash冻结(R05); ST源头排除(R06); 分regime(R11); 扣[动量+市场]消融(R12++); 单月outlier
- **短线sleeve铁律: 必含真实成本(高换手会被吃); 必有Arm_B对照隔离"席位技能"vs"龙虎榜效应"; 必报容量**

## 迭代日志
<!-- 每轮 append -->
- (init) 由 hidden-alpha(已完结)转入。SEAT轨挖到真信号但载体错, 本轮用正确r5事件sleeve冶炼。等待启动。
