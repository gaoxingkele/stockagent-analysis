# Progress — research/hidden-alpha

> fresh-context 每轮干净重启, 跨迭代唯一记忆 (SIGN-R09)。每轮结束前更新。

## 北极星
找隐藏/结构 alpha (非又一个价量因子, 见 [[reference_orthogonal_data_map_0605]] 价量已榨干)。**两轨**:
- **② 概念网络 lead-lag**: 用现成概念库, 同概念/链条**先动票预测滞后票**(结构 alpha, 无新数据)。
- **③ 龙虎榜席位印记**: top_inst 的 `exalter` 席位**历史胜率 → 聪明席位跟随**(行为 alpha)。
各自 Phase-1 廉价筛查(扣[自身动量+市场]后正交IC), 有残差才上 walk-forward gate vs V12.31。
**先验保守(10连否)但这俩是结构/行为信号未必被动量吃; 关键=扣动量后还剩不剩。生产线 V12.31 冻结。**

## 事前注册 gate (冻结 R01 + 分regime R11 + 消融 R12++)
- Phase-1: ≥36月 横截面 rank-IC 扣[自身动量+市场]残差 |IC|>=0.02 且 |t|>=3 → residual_signal, 否则廉价 REJECT。
- GATE: Δα(信号臂−V12.31)>=+0.30pp 且 Sharpe/最差月不降 且 单月outlier剔后>0 且 分regime不伤 且 扣[动量+市场]存活。

## 应对手册 (responsePlaybook, 冻结)
- Phase-1: residual_signal→GATE; no_residual→该轨廉价REJECT记录(另一轨还在)。
- GATE: PASS→落地; 真小(0~+0.3)→记 blend 候选不强上; REJECT→文档化, 生产线不动。
- 防泄漏: 龙虎榜收盘后公告→信号 D+1 生效; 席位胜率只用过去 expanding; 概念lead 只用 ≤t-1 邻居收益。

## 任务台账 (双轨 + 共享 gate)
| id | 任务 | 状态 | 裁决 |
|----|------|------|------|
| NET-001 | 概念网络 lead-lag 特征 | ✅ built | 5.3M行×6特征落盘, 因果loo, 零泄漏, 确定性✓ |
| NET-002 | 概念 Phase-1 廉价筛查 | ✅ REJECT | no_residual: 扣[动量+市场beta]后最强|IC|=0.0177 t=0.97 ≪线, lead-lag≈动量换皮 |
| SEAT-001 | 席位印记特征 (top_inst exalter) | ✅ built | 46K行×5特征, 9235买方席位(2722可信), 因果expanding+D+1, 零泄漏, 确定性✓ |
| SEAT-002 | 席位 Phase-1 廉价筛查 | todo | — |
| GATE-001 | walk-forward gate (有残差的轨) | todo | — |
| LAND-001 | opt-in 落地 (依赖 GATE=PASS) | skip | — |

## 数据资产 (已确认)
- 概念: output/tushare_cache/concept_{detail,list,member_summary}.parquet + output/concept_local/{dc,ths}
- 席位: output/tushare_cache/top_inst.parquet (633613行 2023-01~2026-05, 列 trade_date/ts_code/**exalter(席位名)**/buy/sell/net_buy/side/reason) + top_list.parquet
- daily cache (前向r5/r20 + 自身动量); RG-001 regime_timeline (分层); t005 walk-forward harness; verify.py(isGate)

## 关键约束
- 负结果=合法完成禁同段OOS重调(R02); IC≠落地只认walk-forward α(R03); 泄漏前置闸(R04)
- 生产hash冻结(R05); ST源头排除(R06); 分regime(R11); 扣[动量+市场]消融(R12++); 单月outlier
- **本轨头号泄漏坑: 概念lead只用过去邻居收益(非同期); 龙虎榜D+1生效; 席位胜率只用过去**

## 迭代日志
<!-- 每轮 append -->
- (init) 由 fundamental-orthogonal(已完结 C_塌缩 REJECT)转入。用户选 ②概念网络 + ③席位印记两轨先做。等待启动。
- **NET-001 (06-06) built**: `research/net001_build_leadlag.py` → `research/features/concept_leadlag.parquet`。
  - 设计: 对每(ts_code,t)算同概念邻居 leave-one-out 滞后收益。邻居信号在 d 收盘可知(trailing trail1/5/20),赋给 next_trading_day(d)=t,保证 feature@t 只含 ≤t-1 + 剔本股自身 → 双重防泄漏(同期+自身)。
  - 6 特征: cl_pr1/cl_pr5/cl_pr20 (邻居loo过去N日收益均值, 跨概念平均), cl_lead_ratio (邻居过去5d上涨比例=先动比例), cl_nbr_count, cl_n_concepts。
  - 规模过滤: 概念成员数 [5,500], 剔沪深300/融资融券等指数代理 (lead-lag≈市场beta)。1878概念。
  - 产出: 5,295,652 行 / 5432 股 / 1066 日 (2022-01~2026-06)。非空率: pr1=1.0, pr5/lead_ratio=0.996, pr20=0.98。确定性✓(单日重算比对), 零泄漏✓(verify leakage guard 0违规)。月块checkpoint research/cache/net001/。
  - 下一步 → **NET-002**: Phase-1 廉价筛查, 对 cl_* 信号(尤其 cl_pr5/cl_lead_ratio) 算 ≥36月横截面 rank-IC(前向r20), 扣[自身动量mom_5/20/60 + 市场]残差化, 分regime。核心问: 滞后票扣自身动量后还跟不跟领先票。residual_signal→GATE; no_residual→该轨REJECT(SEAT轨还在)。
- **NET-002 (06-06) no_residual / REJECT**: `research/net002_phase1_screen.py` → `research/cache/net002_{ic_table,panel,beta}.parquet` + `verdicts/NET-002.json`。
  - 口径: 月末 rebalance (mom 缓存 54 月末日, 47 个有前向 r20 → ≥36月功率达标), 逐月横截面 Spearman(信号, fwd r20) 跨月平均 + t=mean/(std/√n)。复刻 FU-003 框架。
  - 控制 (R12++): mom_5/20/60 (自身动量, 复用 fu002_momentum) + beta (市场: per-stock trailing 60d vs 等权市场日收益, 因果, 自算落 net002_beta.parquet)。变体 raw/minus_mom/minus_market/orth_full。
  - **结果**: 4 信号 (cl_pr1/pr5/pr20/lead_ratio) 全期 orth_full(扣动量+市场) 残差 IC: cl_pr1 -0.0177(t-0.97 最强), cl_pr5 -0.0015(t-0.08), cl_pr20 -0.0011, lead_ratio -0.0021 — **全部远未过 |IC|>=0.02 & |t|>=3**。
  - **定位 (消融)**: cl_pr5 raw IC=-0.0023(≈0) → 扣动量 +0.0152(t=0.70, 仍不显著) → 扣市场 -0.0199 → 全扣 -0.0015。即"扣自身动量后微弱翻正但远不显著"⇒ lead-lag 信号 ≈ 自身动量/市场 beta 换皮, **无独立残差**。
  - **裁决 no_residual (SIGN-R02 负结果=合法完成)**: 滞后票扣掉自身动量后并不独立跟随同概念领先票 (A股同概念共涨主要是 beta+动量, 非可交易的领先-滞后结构)。该轨 ② 概念网络 **廉价 REJECT**, 不进 GATE walk-forward。
  - 第 11 个被脚手架否的假设。**SEAT 轨仍在** (③ 龙虎榜席位印记)。GATE-001 是否跑 walk-forward 现完全取决于 SEAT-002: 若 SEAT-002 也 no_residual → GATE-001 status=REJECT 不跑。
  - 下一步 → **SEAT-001**: 龙虎榜席位印记特征 (top_inst exalter 历史 expanding 胜率→聪明席位跟随, 龙虎榜收盘后 D+1 生效防泄漏)。数据 output/tushare_cache/top_inst.parquet (633613行, 列 trade_date/ts_code/exalter/buy/sell/net_buy/side/reason)。
- **SEAT-001 (06-06) built**: `research/seat001_build_footprint.py` → `research/features/seat_footprint.parquet`。
  - 设计: top_inst 每 (date,code,exalter) 去重 (side 0/1 净买入重复) → 买方席位=net_buy>0。每席位历史 expanding 战绩 = 过去净买入事件的 fwd_r5/r20 均值 + r5 胜率, **跨股累计** (席位技能跨标的)。
  - 因果三重防泄漏: ① 席位某历史事件 e 的 outcome 仅当 avail(e)=idx(e)+H ≤ idx(d) (已兑现) 才计入; ② 当前事件自身 avail=idx(d)+H>idx(d) 天然排除 (时间维 leave-one-out); ③ 信号赋 next(d)=D+1 (龙虎榜收盘后公告)。searchsorted 实现 expanding 战绩。
  - 5 特征: sf_edge_r5/sf_edge_r20 (买方席位历史 fwd 收益 net_buy 加权), sf_winrate_r5, sf_n_seats, sf_seat_nhist。MIN_HIST=5 兑现历史才信任席位。fwd_r5/r20 仅落 cache (research/cache/seat001/), **绝不**进 features (verify forward 黑名单)。
  - 产出: 46,057 行 / 4704 股 / 812 日 (2023-01~2026-05); 9235 买方席位 (2722 有 ≥5 兑现历史可信)。非空率 sf_edge_r5=1.0/sf_edge_r20=0.985/winrate=1.0。确定性✓(抽样席位重算比对), 零泄漏✓, ST 源头排除。
  - **观察 (非裁决, 留给 SEAT-002)**: sf_edge_r5 均值 -0.0102 / sf_winrate_r5 均值 0.38 — 席位历史买入**平均跑输** (A股龙虎榜买方多为追涨接盘?)。但这是无符号描述; 横截面 rank-IC 才看"高战绩席位是否真预测高收益", 且 SEAT-002 必扣[自身动量+市场]。
  - 下一步 → **SEAT-002**: Phase-1 廉价筛查。对 sf_edge_r5/r20/winrate 算 ≥30月横截面 rank-IC (r5/r20 都看, 龙虎榜偏短线), 扣[自身动量 mom_5/20/60 + 市场 beta] 残差化, 分 regime。核心问: 扣自身动量后, 跟"聪明席位"还有没有独立残差。residual_signal→GATE; no_residual→SEAT 轨也 REJECT (则 NET+SEAT 双否 → GATE-001=REJECT 不跑 walk-forward)。注: 信号天然稀疏 (仅上榜股 D+1 有值, 46K 行), 月末 rebalance 对齐时横截面会偏小, SEAT-002 需考虑用事件对齐 (上榜 D+1 起 r5/r20) 而非月末对齐, 以保功率 ≥30月。
