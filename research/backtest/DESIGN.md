# Book 级回测引擎 + 因子归因 — 设计文档

> 起因 (2026-06-14): 18 个选股因子实验全否, V12.31 选股近最优。但我们**从没建过真实组合回测器** ——
> 一直只测"等权 Top-N 选股 α"(t005), 没有持仓/sizing/暴露/成本/回撤/换手。用户要它**发现问题**:
> 混合因子里哪些在**实盘 P&L 层面拖后腿**。这是 book/风控层的地基, 服务 (a) 判断 V12.31 该不该真金白银跑
> (b) 后续 RL-lite 风控策略搜索的 environment (c) 用户事件序列洞察的 meta-labeling 落点。
> 生产线 V12.31 (v12_scoring.py / v12_dual_track.py) **全程只读冻结**; 消融在 research 侧做。

## 1. 与现有 t005 的区别 (为什么必须新建)

| | t005 walk-forward (现有) | BT book 回测器 (本设计) |
|---|---|---|
| 评估对象 | 选股 α (等权 Top-N, 月度截面) | 真实组合 P&L 路径 |
| 持仓 | 无 (每期独立) | 持仓 carryover + 换手 |
| 成本 | 无 | A股真实成本 + T+1 |
| sizing | 等权 | 可参数化 (等权 / score 加权 / 双轨) |
| 输出 | α/Sharpe(截面) | 净值曲线/年化/Sharpe/最大回撤/换手/暴露 |
| 用途 | 因子是否有 α | 组合能不能赚、谁在拖后腿 |

## 2. A股成本模型 (预注册冻结, R01)

- 佣金 0.025% 双向 (min 5 元忽略); 印花税 **0.05% 仅卖出** (2023-08-28 后); 过户费 0.001% 双向。
- 滑点基线 **0.10% 每边** (小盘为主, 偏保守); 报 0/0.05/0.10/0.20% 敏感性。
- **T+1**: 买入当日不可卖; 涨跌停不可成交 (近似: |涨幅|≥9.8% 当日不开新仓/不平仓)。
- 整手 100 股: 忽略 (近似连续)。
- round-trip ≈ 0.30% (含 0.10% 滑点) → 月度换手下显著。

## 3. Book 基线 = V12.31 生产规则 (只读复刻)

baseline arm 复刻 `v12_dual_track.py` 生产 book (双轨 70%集中/20%分散/10%现金, 行业 cap 4, 持有~r20≈20d/再平衡周期按生产)。
**复刻 = research 侧调用/重述生产逻辑, 不改 v12_scoring.py/v12_dual_track.py (R05 指纹冻结)。**
sanity: 成本关 + 等权时, BT 数应落在 t005 选股 α 同量级。

## 4. 因子/铁律归因 (用户的"谁拖后腿")

对 V12.31 评分各组件 / V7c 6 铁律逐个 **leave-one-out**, 跑消融 arm 过引擎, 测 book 层 ΔP&L:
- 组件: r20 池模型 / pump_up / pump_down(ratio 分母) / 行业 cap / 双轨 sizing。
- 6 铁律: r20 top5% / pyr_velocity<p35 / 双静默 / 非僵尸 / 行业动量≥0.10。
- 每个报 Δ(净 Sharpe / 年化 / 最大回撤), 分 regime → **标出正贡献 vs 负贡献(拖后腿)**。
- 消融全在 research 侧 (基于 score_market 暴露列重组 / 重述规则), v12_scoring 冻结。

## 5. 预注册第 2 步 (BT-004, skip 待回测器出来再开) — 事件上下文 meta-label 残差测

用户"MA 事件序列/条件上下文"洞察的**最小单表示残差测**(防组合过拟合, López de Prado):
- **单一表示**: 条件上下文 `ctx = (MA20 在事件前 K 日内已上拐?)` × `MA5 上穿 MA20` 事件; 不搜窗口/不搜 MA 组合。
- 测: ctx 对 T+20 的残差 RankIC, 扣 [pump 分数 + MA20斜率 + pyr_velocity + ADX]; 报 **Deflated Sharpe / PBO**。
- gate: 残差 |IC|≥0.01 & t≥2 & 过 deflation → 作 **meta-label / 条件 sizing** 进 book 层 (非新 α 因子); 否则 REJECT_reskin。
- **先验**: 大概率被吃 (关系张量 TCN 已 ≈0), 但这是唯一干净判它的方式; 严格单表示, 不钓鱼。

## 6. 顺序

BT-001 引擎 → BT-002 V12.31 真实净 P&L (该不该交易) → BT-003 因子归因 (谁拖后腿) → BT-004 事件上下文 meta-label = **REJECT_reskin (第19否, MA事件序列被现有TA吃掉)**。
生产线 V12.31 冻结。关联: `[[feedback_quant_system_meta_lessons_0524]]` · `[[project_duokongk_reskin_reject_0614]]` · `[[project_bt004_event_context_metalabel_reject_0614]]`。

---

## 7. EX 阶段 — 出场策略 + 基金对照 (2026-06-14, 承用户 triple-barrier 直觉)

19× 价格/形态选股信号穷尽 → 增量不在新信号, 在**目标函数 / 出场策略 / universe**。承用户"不限天数, 限幅度(分批止盈)+回撤(止损)提 Sharpe"= López de Prado triple-barrier。**拆 3 杠杆, 先验差别巨大**:
- **回撤止损**: ⚠ 强负先验 (`[[project_three_rejects_meanreversion_meta_0603]]` 止损在均值回归上反指, 砍在反弹前)。
- **分批止盈**: 可能正 (遇强减仓顺均值回归 edge)。
- **r20 triple-barrier 重标**: 改选股目标 (偏好干净路径), 不被止损否决覆盖, 直接瞄 Sharpe, 重活。

### EX-001 出场策略测 (现成引擎, 无需重训; 受控 Δ 同 picks 共模相消可信)

对**同一批 V12.31 picks** 套不同出场, 比 book Sharpe/年化/maxDD。**预注册小网格 (不搜, R01)**:
- baseline: 现行固定 ~r20(≈20d)持有。
- 分批止盈 TP: 三档减仓 +10%/+20%/+30% (各 1/3); 触发用日内 high 达标价成交 (缺口越过→开盘价; 涨停不可卖)。
- 回撤止损 SL: 自入场峰值回撤 {−8%, −12%} 触发; 用日内 low; 跌停不可卖。
- 组合 TP+SL; 时间 backstop 40d (去固定 20d 后兜底)。
- **分离报 TP-only / SL-only / 组合 vs baseline** → 验证"止盈帮/止损伤"。close-based 作保守界参考。

### EX-002 基金/风格对照 (量化"基金更强"是选股还是风格)

- V12.31 picks 的**风格画像**: 市值/波动/换手/行业 vs 全市场。
- 对照 book Sharpe: vs 大盘质量基准 (基金风格代理) + 池 C 基金重仓静态篮 (标注幸存者偏差); 若 Tushare fund_portfolio 历史持仓可得则拉真共识篮, 否则降级用静态篮+风格基准并文档化限制。
- 结论: 与基金的 Sharpe 差是**风格/universe (小盘高波→需 tilt)** 还是选股 (可重标修)。

### EX-003 [预注册, 后开] r20 triple-barrier 重标

- label: 不固定天数, 双屏障 (上 +X% 先触 vs 下 −Y% 先触) + 时间 backstop; 目标=P(先触上屏障) 或 屏障路径风险调整收益。
- 屏障参数 (X/Y) **预注册冻结**防过拟合; **book 层 walk-forward apples-to-apples**, 两臂同出场规则防循环自证 (栽过, 见跨horizon)。
- gate: r20_triplebarrier 重训臂 book 净 Sharpe > 原 r20 臂, maxDD 不升, 单月outlier+分regime, de-lookahead 真 walk-forward。
- 仅 EX-001/002 informs 后开。
