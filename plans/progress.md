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
| SL-001 | r5席位sleeve回测引擎(D+1+席位排序+成本+三臂对照) | built | 引擎落盘,三臂Arm_A>Arm_B>Arm_M |
| SL-002 | 成本×持有期×容量敏感性 | built | 网格全档net负(原始均值回归非市场相对α); 容量min-binding中位仅0.054亿小 |
| SL-003 | walk-forward gate(net-of-cost α+席位技能净增量) | DONE | **REJECT_负**: 净α-2.0pp/月 t-6.2; 但席位技能净增量+1.02pp t4.4真 |
| SL-004 | 独立sleeve日频输出(仅PASS) | skip | — (REJECT→保持skip) |

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
- (r1 SL-001=built) 造好 r5 事件 sleeve 回测引擎 `research/sl001_sleeve_engine.py` (D龙虎榜→D+1开盘建仓→D+1+5开盘平仓, 三臂 apples)。817 entry日 (20230104~20260525), 23978全龙虎榜候选, 席位可信覆盖**98.9%** (对照GATE-001里r20池仅3.7%稀疏=载体确实换对了)。**主测描述统计 (H=5/N=10/30bps往返)**: 毛/笔 Arm_M=-1.82% / Arm_A(席位sf_edge_r5)=-1.43% / Arm_B(动量)=-2.47%; 净/笔 A=-1.73%; **Arm_A−Arm_B 净/笔 +1.04pp** (席位技能净增量初窥,gate在SL-003); 年化换手≈50轮→成本拖累≈15%/yr。
  - 关键观察: 龙虎榜净买入股 r5 **原始**收益全为负 (均值回归), Arm_A 只是"最不差"。这是原始收益非市场相对α — SL-003 才算 net-of-cost α(Arm_A−等权市场) + Arm_A vs Arm_B 隔离。但**席位排序 > 动量排序 > 等权全集**的次序已支持"席位技能真实"假设。
  - 产出: `research/cache/sl001_daily_pnl.parquet` (entry×arm gross/net/n) + `research/cache/sl001/picks.parquet` (stock级,含entry_amount供SL-002容量/SL-003 gate复用) + checkpoint px/cand/picks。
  - 下一步 SL-002: 成本0/15/30 × H=3/5/10 网格 + 基于 entry_amount 的容量估算。注意短线sleeve成本拖累已显著(~15%/yr),容量看上榜股D+1成交额。
- (r2 SL-002=built) 造好成本×持有期×容量敏感性 `research/sl002_sensitivity.py` (复用 SL-001 引擎/常数, 仅扫 cost/H, SIGN-R01)。网格 `research/cache/sl002_grid.parquet` 9 cells: cost{0,15,30}bps往返 × H{3,5,10}, 各 H picks 按 H checkpoint (picks_H{3,5,10}.parquet, 812 entry日/41月)。
  - **网格结论 (Arm_A net, 月度=事件等权当月篮子均值, Sharpe=月序列mean/std×√12)**: 全 9 档 net 月度收益**全为负** (-0.66% ~ -2.46%), Sharpe(年) -1.16 ~ -1.87, 正月%21.9~29.3%。这是龙虎榜净买入股 r5 **原始**均值回归 (同 SL-001 观察), **非市场相对α** — 净α(Arm_A−等权市场) 与席位净增量(Arm_A−Arm_B) 才是裁决量, 在 SL-003 walk-forward (SIGN-R03 中间指标≠落地)。
  - **成本/持有期纹理**: 持有期越短 net/笔越小但年换手越高 (H=3→84轮 cost拖累25.2%/yr @30bps; H=5→50轮 15.12%; H=10→25轮 7.56%); 成本 0→15→30bps 单调侵蚀每档约 -0.15pp/笔。短线 sleeve 成本拖累确实是核心约束。
  - **容量 (≤event_date成交额1%/等权top-10/无冲击)**: min-binding (等权受当日最小成交额成分股约束) 中位仅 **0.054亿元** (p25 0.023/p75 0.131) = 540万元, **很小**; sum 松上界中位 0.912亿元。短线高换手日内反复进出会进一步压缩。容量小已文档化 (AC 要求, 非硬gate)。amount 单位=千元×1000=元已校准。
  - 产出: `research/cache/sl002_grid.parquet` + `research/cache/sl001/picks_H{3,5,10}.parquet` + `research/verdicts/SL-002.json` (status=built)。
  - 下一步 SL-003 (gate): ≥30月 walk-forward 事件对齐, 主测30bps/H=5。净月度 α = Arm_A 篮子收益 − 等权市场(同日全市场或全龙虎榜?需定口径,建议等权全市场 r5)。席位净增量 = Arm_A − Arm_B (动量对照, 隔离席位技能 vs 龙虎榜效应)。分regime(R11) + 单月outlier剔除 + 扣[动量+市场]存活。断言 preRegisteredGate。**注意全档原始 net 为负, 真正的问题是相对等权市场是否还有正 α (均值回归股可能跑赢同样下跌的市场)**。playbook 五分叉, REJECT 合法完成(R02)。
- (r3 SL-003=DONE 裁决=REJECT_负) 造好 walk-forward 决策 gate `research/sl003_walkforward_gate.py` (复用 SL-001 引擎/常数, SIGN-R01)。市场基准口径定为**等权全市场 H=5 open→open 前向收益** (ST 源头排除, 按 entry_date), 落 `research/cache/sl001/market_bench_H5.parquet`。逐月 41 月事件对齐 walk-forward (席位战绩 expanding 因果 → 逐月 α 天然 walk-forward), 落 `research/cache/sl003_monthly.parquet`。
  - **裁决 = REJECT_负** (playbook 分叉, SIGN-R02 合法完成): 净α(Arm_A net − 等权全市场) = **-2.006pp/月 t=-6.206** Sharpe(年)-3.36 正月17.1% 最差月-5.54pp; 毛α也=-1.706pp t=-5.28 → 不是成本吃掉 (REJECT_成本吃掉 要求毛α>0), 是**本就跑输市场**。决策树: a_mean≤0 且 g_mean≤0 → REJECT_负。
  - **关键正向发现 (写进裁决但不改 ship): 席位技能净增量 Arm_A − Arm_B (扣动量, cost 抵消) = +1.020pp/月 t=4.433 正月75.6% 显著为真。** 印证 SEAT-002 席位印记是真信号 — 在同一龙虎榜可信池内, 按席位历史战绩(sf_edge_r5)排序 **确实跑赢** 按动量排序(Arm_B)。即"席位技能"被 Arm_B 对照成功隔离, 不是龙虎榜效应 (否则会落 REJECT_龙虎榜效应)。
  - **但这救不了 sleeve**: 整个龙虎榜净买入股 r5 universe 相对等权市场是**净输家** (均值回归), 席位只是把"最不差"的子集挑出来, 绝对水平仍 -2pp/月 远低于市场。单月 outlier 剔 (202410 +3.80pp) 后净α -2.152pp **仍负**; 三 regime (mixed -1.58 / momentum -0.94 / reversal -1.57) 净α**全负**不集中也无救; 全部 6 项 gate 仅 2 项过 (|t|≥3 ✓ + 席位技能显著 ✓), magnitude/sharpe/outlier/regime 全✗。
  - **元结论 (信号真 + 载体再次错)**: SEAT 轨两次 REJECT 但原因不同 — GATE-001 接 V12.31 r20 池 REJECT = horizon/稀疏载体错; 本轮换成正确 r5 短线事件载体, 席位技能本身(相对排序)**确实兑现** (+1.02pp t4.4), 但**事件池(龙虎榜净买入股)整体是负 alpha universe**, 任何只在池内排序的 sleeve 都跑输市场。要用席位印记必须**跳出"只买龙虎榜上榜股"** — 把席位战绩当 cross-sectional 因子作用于更广 universe (但那回到 V12.31 r20 池已 REJECT 的 horizon 问题), 或做多空 (买高战绩席位 / 空低战绩席位 同在龙虎榜内, Arm_A−Arm_B 的 +1.02pp 是 long-short 雏形, 但容量极小+成本高)。本轮 sleeve 路径走完, 干净 REJECT。
  - 产出: `research/sl003_walkforward_gate.py` + `research/cache/sl003_monthly.parquet` + `research/cache/sl001/market_bench_H5.parquet` + `research/verdicts/SL-003.json` (status=REJECT_负 + 全 gate 指标)。SL-004 保持 skip (仅 PASS 解除)。
  - **全部非 skip task (SL-001/002/003) 均有裁决, verify.py 退出 0。北极星问题已回答: 席位印记真信号 (席位技能净增量 +1.02pp t4.4) 但龙虎榜事件池是负α universe → sleeve REJECT_负。**
