# 三原型 Sleeve 架构 — 设计文档

> 起因 (2026-06-13): 实验 15 (pump-debias) 证明「一个 ratio 评分塞三种风险结构不同的启动」必然顾此失彼
> (修 flat-base 盲点 → +0.16pp 但灾难月 −0.12→−1.43)。结论: **偏好是 feature 不是 bug** —— past_r5
> 偏好承担隐性尾部过滤。正路不是改一个 label, 是**按启动原型拆 sleeve**, 各自风控, book 层合并。
> 生产线 V12.31 全程冻结, 本设计是平行候选, 过最终 walk-forward gate 才考虑替换/增强。

## 0. 经验锚点 (为何这么设计)

- **PL-001 杠铃诊断**: pump 模型 P_up 排名 — 强涨(动量) **0.724** > 深跌(深蹲) **0.595** > **flat-mid(横盘) 0.419 ←唯一被低估**。
  即「只认深蹲」是错觉; 真相是**两端都吃、独漏中间横盘 base**。
- **系统 DNA 偏抄底不在启动子 label, 在 V7c 池子过滤** (`pyr_velocity_20_60<p35` 把动量名字筛掉)。
- **CB-003**: 横盘几何 (MA20走平/squeeze/缩量) 是 reskin, 被现有 TA 吃掉 → BASE sleeve 的 α **不靠新几何因子**,
  靠**不同 label + 不同风控**捞回那 +0.16pp 真信号且不付尾部代价。
- **RG-002**: regime 择时已否过一次 (动量态之后均值回归反而 +0.378pp) → regime tilt 必须单独 gate, 不默认。

## 1. 架构 (router → per-sleeve 启动子 → per-sleeve 风控 → book 合并)

```
全市场 ─→ 路由器 ─┬─ SQUAT 启动子 (=现 v3c, 冻结) ──┐
   (原型分桶)     ├─ BASE  启动子 (新, 紧尾 label) ──┼→ cross-sleeve 校准 ─→ book(配额+行业cap)─→ Top N
                  └─ MOMENTUM 启动子 (新, 去速度过滤)─┘     (同一单位可比)
```

## 2. 路由器 (Layer 0) — 预注册的清晰边界 (先硬切, 后期可软隶属)

| 原型 | 判据 (现成因子, 仅分桶不产α) |
|---|---|
| SQUAT 深蹲 | `past_r5 < -0.05` (主), 辅: 距20日高点回撤深 + RSI 低位 |
| BASE 横盘 | `abs(ma20_slope_20d) < flat_thr` + `boll_bw_pctile < 0.30` (squeeze) + 整理时长 ≥ 15日 + `past_r5 ∈ [-0.05, +0.05]` |
| MOMENTUM 动量 | `past_r5 > +0.05` + MA 多头排列 + `pyr_velocity_20_60 高分位` (=被 V7c 过滤那批) |
| NONE | 不入任何 sleeve |

> 阈值 (flat_thr / squeeze 分位 / 时长) 在 SLV-001 预注册冻结, 不在结果上调。

## 3. 三启动子 (Layer 1) — 关键在各自的 label

| sleeve | label (正样本) | 失败模式 | 特征侧重 | α 论点 |
|---|---|---|---|---|
| **SQUAT** | 前向5d max_gain≥10% & max_dd≥−5% (现 v3c) | — | 现 247 生产特征 | 已局部最优, **冻结不动** |
| **BASE** | 放量收盘站上箱顶 & 前向 max_gain≥10% & **更紧 max_dd≥−4% / 更短窗** | **假突破** (突破后秒回, 左尾肥) | 箱体高度/时长/缩量 + **突破触发**(量能/缺口/close vs 箱顶) | 捞回实验15的 +0.16pp 真信号, **用紧 dd label 替代被去掉的 past_r5 隐性尾控** |
| **MOMENTUM** | 前向延续且**不见顶** | **力竭/见顶回落** | 去 velocity 过滤 + **力竭守卫**(RSI极值/放量见顶/偏离MA过远 → 负样本) | 标量 α 可能已 price-in; 价值在 **book 对冲**(squat 流血的动量月反向赚) |

## 4. Per-sleeve 风控 (Layer 2) — 拆 sleeve 的物理原因

| sleeve | 持有 | 止损 | 仓位 |
|---|---|---|---|
| SQUAT | ~20d | 偏宽 (均值回归容噪) | 标准 |
| BASE | 短 | **紧贴箱体下沿** (假突破即走) | **小** (质量低) |
| MOMENTUM | 趋势跟踪 | 移动止损 / 力竭即出 | 让赢家跑 |

## 5. 适配评分系统 (Layer 3) — 三条并成一个推荐表

- **B (主) cross-sleeve 校准 + book 配额** ✅: 每条 sleeve 分数各自 isotonic **校准到同一单位**(已实现前向风险调整收益)
  → BASE 的 0.7 与 SQUAT 的 0.7 期望结果相同 → 解决实验15「横盘 ratio 不公平抢名额拖尾」。合并成一张排序表,
  加 **per-sleeve 配额**(永远持一部分每种=跨原型分散) + 现有行业 cap 4。排序用**校准后风险调整分**, 非裸 ratio。
- **A (副, 单独 gate) regime tilt**: 动量态倾斜 MOMENTUM / 反转态倾斜 SQUAT。**必须独立预注册 gate** (RG-002 前科), 过不了只留 B。

## 6. 评估纪律 (3×参数 = 3×过拟合面, gate 更死)

- 每 sleeve 预注册冻结 gate (R01), walk-forward apples-to-apples (复用 `research/t005_walk_forward_gate.py`), negative=合法完成 (R02)。
- **BASE kill 标准**: blend(squat+base) **Δα>0 AND 最差月不比 squat-alone 差**。最差月仍恶化 → 假设证伪(横盘启动就是质量低), 诚实接受。
- **MOMENTUM 在 book 层评**: 加它后 book Sharpe↑ AND 动量月最差月改善, 否则不留。
- 生产线 V12.31 冻结 (verify.py 指纹), 三 sleeve 是平行候选。

## 7. 分阶段 (先证伪最便宜的)

| Phase | 内容 | 退出闸 |
|---|---|---|
| **0 (本轮, 零训练几天)** | 路由器 + 三原型机会量化 + **动量是否在深蹲最差月反向**廉价闸 | 反相关不成立 → MOMENTUM sleeve 不建 |
| 1 | BASE 启动子 (紧尾 label) → 单条+blend walk-forward | Δα>0 且最差月不恶化 |
| 2 | MOMENTUM 启动子 (去速度+力竭守卫) → book 层 | book Sharpe/最差月改善 |
| 3 | cross-sleeve 校准 + 配额 + (单独 gate) regime tilt | 3-sleeve book vs V12.31 最终 gate |

## 8. 诚实预期

真正可能赢的是 **book 层 Sharpe / 最差月**(分散 + 各自尾控), **不是裸 α 大涨** —— BASE/MOMENTUM 的 α 可能部分已 price-in,
论点主要押**风险结构**不是新信号。最可能: Phase 1 BASE 过的概率中等; MOMENTUM 取决 Phase 0 反相关检验; regime tilt 大概率仍否。

## 9. Phase 0 结果 (2026-06-13) + Phase 1 修订

Phase 0 零训练筛查 (4.19M stock-day) **把 §3 质量先验几乎全反转**, 架构据此收窄:

- **SLV-001**: 桶内启动率 MOMENTUM 0.206(1.69x) > SQUAT 0.151(1.24x) > **BASE 0.063(0.515x, 反低于市场)**。严格几何 BASE(squeeze+MA20走平+整理≥15日)**很少启动**(呼应 CB-002 coiling<dead)。
- **SLV-002 (先验全 False)**: 20d 质量 **BASE 最高**(μ+0.151/胜率0.829/dd_p10 −0.075 最薄) > SQUAT(μ+0.140/−0.135) > **MOMENTUM 最差**(μ+0.121/dd_p10 −0.159 力竭肥尾)。BASE 不是"假突破肥左尾", 是"**罕见但干净**"(n仅4,987/3.5年)。MOMENTUM 裸 α 最弱(已 price-in)。
- **SLV-003**: MOMENTUM 分散闸 corr(SQUAT,MOM)=0.593≥0.3 → **REJECT**, 砍 MOMENTUM sleeve。(caveat: 动量态 corr 仅 0.069, 留作 Phase 3 regime overlay, 非独立 sleeve。)

**架构收窄: 三 sleeve → 双 sleeve (SQUAT 生产核心 + BASE 小而精 premium)。**

### Phase 1 修订 (BASE sleeve)

- **论点反转**: BASE 不再是"捕一大块漏掉的横盘", 是"**罕见但最干净的突破**"。这**解释 pump-debias 矛盾**: broad flat-mid(−5~5%)拖入低质名字→+0.16 但加深尾; **严格几何 BASE 才是 premium**。
- **v1 优先规则法, 非急于训模型**: SLV-002 证 strict-BASE 入场**本身就是 edge**(无模型已是最高质量) → BASE sleeve v1 = 严格入场+突破确认+简单质量排序, **绕开 n=4,987 小样本训练陷阱**; 训练打分作 v2 仅当样本够。
- **入场 = 严格 router + 突破触发**(放量收盘站上箱顶); **label 紧 dd ≥ −4%**(替代被去掉的 past_r5 隐性尾控)。
- **blend 评估 apples-to-apples**: ArmA=SQUAT-only(=V12.31), ArmB=squat+base(保留 Q 个名额给 BASE 高质入场, 其余 ratio 排序填), 同 N/行业cap。**kill: Δα>0 AND 最差月不比 squat-alone 差**(全论点)。最差月仍恶化→strict premium 太罕见/已priced, 诚实否。
- **隐忧**: BASE 月频候选稀少(可能多日 0 个)→ blend 多数日≈baseline, 偶发注入; 容量/统计功率是真风险, 必须 log 不可静默。

---
**关联**: `[[project_pump_debias_bias_is_feature_0612]]` · `[[project_consolidation_base_blindspot_confirmed_0612]]` ·
`[[project_sleeves_phase0_priors_reversed_0613]]` · `[[project_v3c_momentum_regime_mismatch_0603]]` · `[[feedback_train_label_over_inference_hack]]`
