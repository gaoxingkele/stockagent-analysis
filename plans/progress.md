# Progress — research/meihua-traj

> fresh-context 每轮干净重启, 这里是跨迭代的唯一记忆 (SIGN-R09)。每轮结束前更新。

## 北极星
用户批评单日起卦贫乏(代码=静态ID/日期=当日常数/价格末位=噪声), 提出把'近N日走势'做进梅花。
问题升级为: **梅花对走势的数论分桶能否超过/补充标准技术指标对同一走势的编码**。按顺序检验三方案
**B(体用=趋势vs当日) → A(轨迹数字起卦) → C(累积卦+序列特征)**, 三组对比, **升级 gate**:
残差必须扣除"公历月×板块 **+ 标准动量/趋势因子**"后仍存在。先验=仍偏怀疑但**这次是真问题非结构性必死**
(轨迹是个股特异+随时间变, 有信息可编码)。**生产线 V12.31 冻结。**

## 事前注册 gate (冻结, R01 + 分regime R11 + 升级消融 R12+)
- Phase1 筛查: 残差(扣 月×板块 + 动量因子) |IC|>=0.01 且 |t|>=3 → residual_signal。
- Phase2 (有残差才跑): 19月 walk-forward Δα>=+0.30pp 且 Sharpe/最差月不降 且 消融存活。
- **三组全 no_residual → REJECT (轨迹梅花亦无独立于动量信号), 三组对比本身是交付物。**

## 任务台账 (按顺序 B→A→C→对比→落地)
| id | 方案 | 状态 | 裁决 |
|----|------|------|------|
| MT-001 | B 体用=趋势vs当日 | todo | — |
| MT-002 | A 轨迹数字起卦 | todo | — |
| MT-003 | C 累积卦+序列特征 | todo | — |
| MT-004 | 三组对比 + 决策 gate | todo | — |
| MT-005 | opt-in 落地 (依赖 MT-004=PASS) | skip | — |

## 已有资产 / 复用
- 编码器 research/meihua_encoder.py (单日版已建, 本轮扩展轨迹起卦)
- 梅花核心 vendored from ds-oracle-cli; load_daily 给 close/high/low 路径
- RG-001 regime_timeline.parquet (分层); 标准动量因子在 factor_lab / load_window 里 (消融对照用)
- mh002_phase1_screen.py 的 OOF 编码+残差化框架可复用升级 (这次消融多加动量因子)
- t005 walk-forward harness (Phase2 用)

## 关键约束
- 负结果=合法完成, 禁同段OOS重调 (R02); 中间指标(IC)≠落地只认walk-forward α (R03)
- IC/回测/特征前先跑 leakage guard (R04); 生产hash冻结 (R05); ST源头排除 (R06)
- R11 按regime分层; **R12+ 轨迹梅花消融必须扣 月×板块 AND 动量因子 (否则只是动量换皮=自欺)**

## 迭代日志
<!-- 每轮 append: 迭代N | task | 做了什么 | 裁决/产出路径 | 下一步 -->
- (init) 由 meihua 循环 (单日版已 REJECT) 转入。用户批评单日贫乏, 要 B/A/C 轨迹三方案对比。等待启动。
