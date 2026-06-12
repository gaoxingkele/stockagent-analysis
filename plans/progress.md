# Progress — research/consolidation-base

> fresh-context 每轮干净重启, 跨迭代唯一记忆 (SIGN-R09)。每轮结束前更新。

## 北极星 (用户洞察落地)
V12.31 偏好"下蹲后起跳"(past_r5<0), 但**上涨前形态是一族**: 回调 / 宽幅横盘 / 微涨震荡 / 旗形三角楔形向上整理。
**统一特征 = 上涨前 MA20 走平 + 收敛蓄势。** 系统只抓下蹲、漏掉横盘启动; 且第6铁律"横盘僵尸过滤"可能误杀好的蓄势基底。
**三问**: ① MA20平/蓄势候选能否抓到 past_r5≥0 的横盘/三角启动(系统漏的)? ② 僵尸过滤是否误杀蓄势基底?
③ 扣现有形态因子(MA/ADX/布林/past_r5)后还有无独立 walk-forward α? **选择面细化(非被否的出场/择时overlay), 有公平胜算。生产线 V12.31 冻结。**

## 事前注册 gate (冻结 R01 + 分regime R11 + 消融 R12++)
- Phase1: 蓄势信号(MA20走平+收敛+量能) 对前向 r20 扣[动量+市场+现有形态因子] 残差 |IC|>=0.02 且 |t|>=3 → residual。
- gate: Δα(蓄势臂−V12.31)>=+0.30pp 且 Sharpe/最差月不降 且 单月outlier剔后>0 且 分regime不伤 且 扣现有形态因子存活。

## 应对手册 (冻结)
- CB-002 盲点: past_r5≥0启动占比高且系统漏→验证盲点继续; 否则用户假设证伪先验降。
- CB-002 僵尸: 蓄势基底假阴性率高→僵尸过滤需精炼; 否则不动。
- CB-003: residual→CB-004; no_residual→蓄势≈现有形态因子换皮 廉价REJECT。
- CB-004: PASS→改进v7c池落地; 真小→记候选不强上; REJECT→文档化生产不动。

## 任务台账
| id | 任务 | 状态 | 裁决 |
|----|------|------|------|
| CB-001 | 蓄势候选特征(MA20走平+收敛+量能)+僵尸分解 | done | built (见迭代日志) |
| CB-002 | 盲点 & 僵尸误杀 量化(直答前两问) | done | 盲点=存在 / 僵尸=无误杀 |
| CB-003 | Phase-1 廉价筛查(扣现有形态因子正交IC) | done | no_residual (蓄势信号被现有 TA 吸收, 换皮) |
| CB-004 | walk-forward gate(蓄势候选/放松僵尸 三臂) | skip | precondition skip (CB-003 no_residual) |
| CB-005 | 改进v7c池 opt-in 落地(仅PASS) | skip | — |

## 已有资产
- src/stockagent_analysis/zombie_filter.py (僵尸过滤逻辑, 待审计/分解); daily cache OHLCV(MA20/振幅/量能)
- factor_lab (现有形态因子 MA斜率/ADX/布林/rsi = 消融控制); 生产 pump v3c + r20 口径; t005 walk-forward harness; RG-001 regime; verify.py(isGate)
- 上下文: [[project_three_rejects_meanreversion_meta_0603]] (健康回调=唯一非反指的选择面细化, 本轮同类)

## 关键约束
- 负结果=合法完成禁同段OOS重调(R02); IC≠落地只认walk-forward α(R03); 泄漏闸(R04)
- 生产hash冻结(R05); ST源头排除(R06); 分regime(R11); 扣[动量+市场+现有形态因子]消融(R12++); 单月outlier
- **本轮防自欺核心: 蓄势信号别被系统已有的 MA/ADX/布林因子吃掉(换皮); 用 Arm 对照隔离"纳入横盘启动"的净增量**

## 迭代日志
<!-- 每轮 append -->
- (init) 由 seat-sleeve 转入。用户洞察: 启动前形态一族(下蹲只是其一), MA20走平是统一特征; 疑僵尸过滤误杀蓄势基底。等待启动。
- **CB-001 done** (2026-06-12): `research/cb001_build_consolidation.py` → `research/features/consolidation.parquet` (5.33M 行 × 17 列, 5439 股 / 1073 日 / 20220104~20260611)。
  - 特征族 cs_*: MA20走平(cs_ma20_slope/absslope, |slope|<0.02 走平) + 收敛(cs_bw_squeeze/amp_squeeze, /60d基线<1) + 量能(cs_vol_ratio 5d/20d<1 缩量) + cs_past_r5(backward) + cs_is_consolidation(合成候选)。
  - **base_type 分布**: pullback(下蹲) 27.9% / **flat_base(横盘) 17.5%** / **up_drift(微涨) 18.7%** / other 35.5% / na 0.5% → **非下蹲基底(flat+drift) 占 36%**, 初步支持"系统只抓下蹲会漏 1/3 候选"的盲点假设(CB-002 量化)。
  - **僵尸分解**: cs_zombie 454k 拆 cs_zombie_coiling(蓄势横盘=收敛&缩量) 228k / cs_zombie_dead(阴跌死水) 226k → 僵尸里近半是收敛缩量的蓄势态, 第6铁律是否误杀待 CB-002 测假阴性率。
  - 复用 src/.../zombie_filter.compute_zombie_factors; 确定性逐位比对 True; 全 backward 零泄漏(verify 0 violations); ST 源头排除 266 只。checkpoint: research/cache/cb001/。
  - 下一步 **CB-002**: 定义真启动(前向5日 max_gain>=10% & dd>=-5%, 仅作 outcome 测量), 量化 past_r5≥0 启动占比 + V12.31 v7c 池抓到/漏掉 + 被 is_zombie 剔除票的蓄势基底假阴性率。命中 responsePlaybook 分支。
- **CB-002 done** (2026-06-12): `research/cb002_blindspot_zombie.py` → `research/cache/cb002/launch_panel.parquet` (forward outcome, **非 features 目录, 不触发泄漏闸**)。真启动=生产 pump-up s5 口径(H=5, max_gain>=10% & max_dd>=-5%)。评估 5.30M 决策点/643k 真启动(基准启动率 12.14%)。
  - **①盲点=存在** (命中 responsePlaybook CB-002_盲点.存在): 真启动里 **past_r5≥0 占 52.2%** (非下蹲基底 flat_base+up_drift 占启动 32.8%); 关键 — **past_r5≥0 启动率 13.15% > past_r5<0 11.19%**(基准 12.14%), 即系统偏好的下蹲反而启动率略低。各 base_type 启动率: up_drift/flat_base/pullback/other ~ 见 verdict。V12.31 **第6铁律(zombie)硬漏启动 4.2%**, **下蹲软偏好降权启动 52.2%**。分 regime 一致(past_r5≥0 占启动: mixed 56.3%/momentum 48.0%/reversal 55.7%) → 盲点真实, 非某 regime 假象。→ **验证盲点, CB-003/004 继续**。
  - **②僵尸=无误杀** (命中 CB-002_僵尸.无误杀): 僵尸启动率 5.88% vs 非僵尸 12.72%; **coiling(蓄势横盘)启动率 5.0% < dead(死横盘)6.8%** (lift 0.74, <1)。即"收敛+缩量看似好基底"的直觉在**已是僵尸**的子集里不成立 — coiling 僵尸反而更难启动。→ 6铁律没误杀蓄势金矿, **不动它**; CB-004 的 arm_zombie 臂先验降低(仍可测)。
  - 确定性逐位比对 True; ST 源头排除 266 只; forward 仅作 outcome 不入 features(零泄漏)。
  - 下一步 **CB-003**: Phase-1 廉价筛查 — 蓄势信号(ma20_flat/收敛/量能)对前向 r20 的横截面残差 IC, 扣[动量+市场+现有形态因子(MA斜率/ADX/布林/past_r5/rsi)], 分 regime, 逐步消融。residual_signal→CB-004; no_residual→廉价 REJECT。注意盲点虽真实但 CB-003 要防"蓄势信号被现有 MA/ADX/布林因子换皮"。
- **CB-003 done = no_residual** (2026-06-12): `research/cb003_phase1_screen.py` → `research/cache/cb003/` (controls_monthend.parquet + ic_table.parquet; **forward fr20 仅 cache 不进 features**)。月末 rebalance 54月/268k 决策点, 5439 股。新算控制: ADX(14)/RSI(14) Wilder + mom_5/20/60 + 等权市场 beta; PATTERN 里 MA斜率/布林带宽/past_r5 复用 CB-001 cs_*。
  - **裁决 no_residual** (命中 playbook CB-003.no_residual): 最强蓄势信号 s_bw_squeeze orth_full(扣[动量+市场+MA斜率/ADX/布林/past_r5/RSI]) IC=**-0.0273 |t|=2.53** < 阈(|IC|≥0.02 & |t|≥3)。
  - **逐步消融定位换皮**: s_coil(合成蓄势强度) raw IC=+0.0301(t2.26) → 扣动量 +0.016 → 扣市场 +0.038 → **单扣形态因子即翻负 -0.019** → 全扣 -0.011。即蓄势信号的正向 IC 主要靠"低动量+收敛"这些被现有 MA斜率/ADX/布林/past_r5 标准 TA **完全吸收**, 扣掉后不仅归零还反向 → 纯换皮。唯一扛住 minus_pattern 的 s_vol_shrink(缩量) 全扣后也仅 +0.0132/t1.89, 不过线。
  - **关键澄清**: 与 CB-002 盲点真实(past_r5≥0 启动被系统漏)**不矛盾** — 盲点='系统漏掉这族启动'(选择面问题), CB-003='蓄势特征是否提供超越现有因子的增量预测'(因子正交性问题), 二者独立。盲点真实但**用蓄势信号去补这个盲点≈用现有 TA 因子换个名字**, 无增量 alpha。
  - **元观察**: 这是本研究序列第 N 次"价格再编码归约为 V12.31 已有标准 TA"(见 memory project_relation_tensor_reject_0604 元观察)。MA20走平+收敛+缩量本质就是 低|MA斜率|+低布林带宽+低量比, 这些早已是标准 TA 控制项。V12.31 在价量空间是强局部最优, 蓄势形态没逃出这个空间。
  - **CB-004 自动 skip** (脚本写 prd.json skip=true): no_residual 触发 precondition skip, 不跑 walk-forward (SIGN-R03 中间指标不落地 + SIGN-R12++ 消融闸 + playbook CB-003 分叉)。
  - 确定性: 复用 CB-001/CB-002 同源 daily; ST 源头排除 266 只; fr20 forward 仅 outcome。checkpoint research/cache/cb003/。
  - **整轮收尾**: CB-001(built)/CB-002(盲点真+僵尸无误杀)/CB-003(no_residual)/CB-004(skip)/CB-005(skip)。北极星三问已答: ①盲点存在 ②僵尸无误杀 ③扣现有形态因子后无独立 α(蓄势信号换皮)。生产线 V12.31 冻结未动。
