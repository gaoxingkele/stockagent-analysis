# Progress — research/relation-tensor

> fresh-context 每轮干净重启, 这里是跨迭代的唯一记忆 (SIGN-R09)。每轮结束前更新。
> **完整设计见 research/DESIGN_relation_tensor_v2.md (必读)。**

## 北极星
K线相对位置多尺度软符号关系张量 (v2) → CNN+TCN 编码 + 标量因子融合 → 截面打分
(主=r20潜力股, 辅=pump启动子)。**核心待答**: CNN 学到的非线性形态交互能否在扣除标准TA标量版后
仍带可交易增量 Δα。先验=非结构性必死但大头≈标准TA, 真增量小需验证。**生产线 V12.31 冻结。**

## 锁定决策 (2026-06-04)
N=40 锚点 / 深度 5 / κ=1 / **双通道**(硬符号+连续) / **两条 head 都跑对比**(先 GBDT-on-emb 后 NN) /
标签 **r20 为主**(gate 决策)+ pump 辅 / gate **+0.30pp**。

## 事前注册 gate (冻结 R01 + 分regime R11 + 升级消融 R12 + 单月outlier)
- Phase-1 (RT-004, 仅 r20+GBDT-on-emb): A2 embedding 对 r20 残差(扣 A0标量TA+月×板块) |IC|>=0.01 且 |t|>=3 → 继续, 否则廉价 REJECT。
- Phase-2 (RT-005): Δα(A2−A0)>=+0.30pp 且 Sharpe/最差月不降 且 单月outlier剔除后仍>0 且 分regime不伤 且 标准TA消融存活。

## 任务台账
| id | 任务 | 状态 | 裁决 |
|----|------|------|------|
| RT-001 | 关系张量 v2 编码器 | **done** | built (5.19M锚点 落盘) |
| RT-002 | A0 标准TA标量对照 | todo | — |
| RT-003 | CNN+TCN 编码器 + 双head | todo | — |
| RT-004 | Phase-1 廉价筛查 (r20+GBDT-on-emb) | todo | — |
| RT-005 | walk-forward gate 四臂 | todo | — |
| RT-006 | opt-in 落地 (依赖 RT-005=PASS) | skip | — |

## 控算力执行顺序 (设计 §10, 强制)
组合很大(N=40×深度5×4×4×2 ≈6.4K值/样本 × 月度重训 × 双head×四臂×双标签)。强制:
1. 先建编码器+A0 (RT-001/002), 分块落盘 + 自检。
2. Phase-1 (RT-004) 只在 r20+GBDT-on-emb 跑廉价筛查; 无残差直接跳过 walk-forward (省90%算力)。
3. 有残差才进 RT-005, 仍分级: r20×GBDT四臂 → 过了补 NN → 补 pump; 任一级 REJECT 即止。
4. checkpoint 强制 (SIGN-R08): 特征分块/月度模型缓存/已完成跳过。

## 复用资产
- daily cache OHLCV (张量路径); load_window/factor_lab (A0+标签); 生产 r5_pump_3way_lgbm_v3c + r20 口径
- t005 walk-forward harness; RG-001 regime_timeline (分层); verify.py(isGate) + guardrails R01-R12
- 关系张量与梅花轨迹同类陷阱: 大概率≈标准TA换皮 → A2-vs-A0 主对照 + 标准TA消融是核心

## 关键约束
- 负结果=合法完成禁同段OOS重调(R02); IC≠落地只认walk-forward α(R03); 泄漏前置闸(R04)
- 生产hash冻结(R05); ST源头排除(R06); 分regime(R11); 奇异特征扣标准TA消融存活(R12); 单月outlier检验

## 迭代日志
<!-- 每轮 append: 迭代N | task | 做了什么 | 裁决/产出路径 | 下一步 -->
- (init) 由 meihua-traj 循环(已完结REJECT)转入。承用户"K线相对位置张量"设计, spec 已成熟锁定决策。等待启动。
- 迭代1 | RT-001 | 关系张量 v2 编码器落盘。**关键工程决策: per-anchor 存储** —— 一行=一个锚点 (ts_code,trade_date) 存 170 值 (160价4×4×2双通道 + 5量 + 5形态), 序列张量 N=40×5×4×4×2 是模型输入视图, 由 RT-003 data loader 按 ts_code 取尾部40锚点行拼装。语义等价 (锚点本就是(ts_code,date)) 且省 38x 盘 (6.4K→170值/样本, 2.7G vs ~128G)。| 产出: research/rt_encoder.py (核心, 全panel groupby-shift 向量化, 36s/全史) + research/rt001_build_tensor.py (月分块checkpoint) → research/features/rel_tensor_v2/ 53月分块 + _index.parquet; verdict RT-001.json status=built。5.19M锚点×170特征 (2022-01~2026-06, 5435股 ST已源头排除); 确定性=True (200股抽样重算逐位一致) NaN=0% 零泄漏 (锚点t仅用≤t, groupby shift); 软符号 mean=-0.021 std=0.635 饱和3.2% 死区8.7% (κ=1, σ20归一, √d阶归一)。verify.py 无VIOLATION (leakage/fingerprint清), 仅 RT-002..005 待做。| **下一步: RT-002** A0标准TA标量对照特征集 (多尺度动量mom_1/3/5/10/20 + ATR/range + 量比 + 缺口 + 复用生产因子), 与张量同口径同期, 是"扣标准TA消融"核心对照。
