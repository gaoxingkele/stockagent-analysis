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
| MH-002 Phase1 regime分层IC + 朴素消融 | todo | — |
| MH-003 Phase2 梅花特征加进排序模型臂 | todo | — |
| MH-004 walk-forward gate | todo | — |
| MH-005 opt-in 落地 (依赖 MH-004=PASS) | skip | — |

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
