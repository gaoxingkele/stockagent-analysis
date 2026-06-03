# Progress — research/meihua

> fresh-context 每轮干净重启, 这里是跨迭代的唯一记忆 (SIGN-R09)。每轮结束前更新。

## 北极星
把梅花易数起卦 (research/meihua_encoder.py, vendored from ds-oracle-cli) 做成确定性 categorical 特征,
检验它对起涨点 (+20 forward) 是否有经 walk-forward 验证、且**扣除朴素对照 (公历月+板块cohort) 后仍存在**
的真 alpha。两阶段: Phase1 廉价 IC+消融筛查 (无残差则廉价收手), Phase2 walk-forward gate。
**先验=大概率噪声** (组合卦≈日历+价格末位, 静态卦≈股票ID)。**生产线 V12.31 冻结。**

## 事前注册 gate (冻结, SIGN-R01 + 分regime R11 + 消融存活 R12)
19 月 walk-forward, 梅花特征臂 vs 无梅花 baseline:
- Δα ≥ +0.30pp 且 Sharpe/最差月不低于 baseline 且 **增益扣除"公历月+板块cohort"后仍存在** → PASS, 否则 REJECT。

## 任务台账
| id | 状态 | 裁决 |
|----|------|------|
| MH-001 全历史梅花特征 parquet + 校验 | done | built — 5.3M 行特征落盘, 确定性/零泄漏/分布 sanity 全通过 |
| MH-002 Phase1 regime分层IC + 朴素消融 | done | **no_residual_signal** — 动态 mh_* 残差 \|IC\|<=0.0011 (≪0.01 floor), 卦象被 月×板块 朴素对照解释完毕, 廉价 REJECT |
| MH-003 Phase2 梅花特征加进排序模型臂 | skip | skipped — MH-002 无残差, 不构建梅花排序臂 |
| MH-004 walk-forward gate | skip | skipped — 无残差信号送 gate, gate 从未 PASS |
| MH-005 opt-in 落地 (依赖 MH-004=PASS) | skip | skipped — 未 PASS 不落地, 生产线冻结未动 |

> **循环结论 (2026-06-04): 梅花易数起卦特征对 +20 起涨点无经朴素消融存活的预测力 → 全研究廉价 REJECT, 生产线 V12.31 不接入。先验(噪声)被数据证实。**

## 已有资产
- 编码器 research/meihua_encoder.py 已建并自检通过 (确定性=True, 体用5类18-21%, 64卦全覆盖, 零泄漏)
- 梅花核心 vendored from D:/aicoding/ds-oracle-cli/app/engine/meihua.py
- regime 时间线 research/features/regime_timeline.parquet (RG-001, 分层用)
- t005 walk-forward harness (research/t005_walk_forward_gate.py) + 缓存模型可复用口径

## 关键约束 (摘自 guardrails)
- 负结果=合法完成, 禁止同段 OOS 反复重调 (R02); 中间指标(IC)≠落地只认 walk-forward α (R03)
- 任何 IC/回测/特征前先跑 leakage guard (R04); 生产文件 hash 冻结 (R05); ST 源头排除 (R06)
- R11 评估必须按 regime 分层; **R12 奇异特征必须存活朴素消融 (月历+板块), 否则=自欺**

## 迭代日志
<!-- 每轮 append: 迭代N | task | 做了什么 | 裁决/产出路径 | 下一步 -->
- (init) 由 regime-overlay 循环 (已完结) 转入。用户提供 ds-oracle-cli 梅花算法, 编码器已建。等待启动。
- 迭代1 | MH-001 | research/mh001_build_features.py: 全历史 daily(ts_code,trade_date,close) → encode_frame → research/features/meihua_features.parquet (5,302,085 行 × 20 特征, 20220104~20260603)。ST 源头排除 266 只(280K 行, SIGN-R06)。落盘前泄漏自查 (输出列仅 key+mh_*/mhs_*, 无 forward 字段, SIGN-R04)。确定性自检=True (5w 抽样重算逐位一致)。分布 sanity: 体用关系 5 类不退化 (21.9/18.7/18.8/20.3/20.3%), 组合卦 base 64/64, 静态卦 57/64, 动爻六位均匀。裁决 research/verdicts/MH-001.json status=built。下一步: MH-002 Phase1 廉价筛查 (regime 分层 rank-IC + 公历月+板块cohort 朴素消融, SIGN-R12)。注意: 朴素对照若吃掉梅花增益 → no_residual_signal → MH-003/004 置 skip 廉价收手。
- 迭代2 | MH-002 | research/mh002_phase1_screen.py: 装配分析面板 (梅花特征 + 前向 r20 label[留 cache 不进 features, SIGN-R04] + RG-001 regime + 板块 + 公历月; 5,193,492 行 / 5422 股 / 1047 日)。三层口径: (1) 原始序数码横截面 rank-IC, (2) 月度扩张窗 OOF target encoding (gap=2 月, 给名义卦象 id 公平机会, 零泄漏) 的 IC, (3) 朴素对照 = (公历月×板块) OOF 编码 → naive_pred → 残差 = r20 - naive_pred → 梅花 OOF 编码对残差的独立残差 IC。全期 + 分 regime(momentum/mixed/reversal, SIGN-R11)。**裁决 no_residual_signal**: 所有 10 个动态 mh_* 残差 \|IC\|<=0.0011 (最强 mh_yang_count IC=-0.0011 t=-2.15, |IC| 比 0.01 floor 低一个量级, t 只因 1012 日才过线 = 经济上零信号); 朴素对照本身 OOF-IC=+0.0073。静态 mhs_* 残差 IC 较大 (至 -0.018, t 至 -10) 但扣板块后反增, 证实 = 卦象版股票 ID 的个股固定效应 (per-stock 持续均值), 非占卜信号。关键口径说明: 横截面 IC 天然差掉当日全市场常数项, 纯公历月季节性根本无法进入 IC, 朴素对照仅通过 月×板块 交互起作用。precondition 触发 → MH-003/004 skip:true, MH-005 skipped, 各写 status=skipped verdict。**全梅花研究廉价 REJECT, 生产线不接入, 先验(噪声)证实。** 产出 research/cache/mh002_results.json + mh002_panel.parquet。下一步: 无 (所有 task 已 done/skip)。
