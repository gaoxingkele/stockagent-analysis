# S20 checkpoint — 2026-09-20

状态：诊断策略循环完成。**非正式 H04。未重训冻结排序，未授权正式训练，不进生产。** H01 仍是 FAILED_VALIDITY。生产 R20 / `pool_e` / 网页 / 21:00 未改。

思想条目：`wiki/2026-09-20_s20-causal-screen-specified-rise.md`。

## 发运（当前有效）

π0 = 冻结 multinomial_base Top20，惩罚效用 `pA−1·pB−2·pD−0.25·pC`，独立风险门关闭。

主档评价：**+15% 可提前走 / −10% 路径**。次档 +25%/−15% 只报不选。

| 口径 | 回放 Top20 | 外层 Top20 |
|---|---:|---:|
| 配对盈利 A∪B | 59.4% | **56.3%**（320 只；A 179 / B 1 / C 122 / D 18） |
| 配对 −10% | 3.3% | **5.9%** |
| 指定 +15% 止盈命中 | 16.1% | **13.4%** |

指定上涨 ≠ 到期小涨。外层 56.3%「赚钱」里大部分没摸到 +15%。

## 冻结决策

1. 排序权重仍是 v4-rsi-campaign 的 multinomial_base。按配对四类重训过一次，外层 55.3%/11.9%，未超过冻结排序，否决升级。
2. Wilder RSI 特征否决（打乱对照不差于真 RSI）。Dream-RSI / Harness-RSI 只进化选择规则。
3. P(B10) 必须 `label_available_at < prediction_at` 滚动 Platt 后再卡门；未校准分不当绝对概率。
4. 门槛/网格只在回放段用分数质量登记，外层只确认。覆盖分母钉死 π0 的 20×信号日，≥0.3。
5. 沉默 = 窗口最高涨幅 < 8% 且未破 −10%。**不是脏数据。** 当天未走完的最高价不得剔当天票。
6. 高上涨分不能抵消独立 −10% 门。不承诺不亏损，不把 G4/G5 当训练暗门。

## 本轮做过、没有发运的挑战者

- 校准后 P(B10) τ=0.26：回放盈利 60.6%、回撤持平 3.3%；外层与 π0 相同，未严格再改善。
- 外层 τ=0.20 的 56.6%/5.6% **不得当选点**（回放覆盖 8%）。
- 事后删沉默：外层盈利可到 ~75–77%，−10% 升到 ~15–17%。换总体，双约束失败。
- 成熟沉默冷却：指定 +15% 略升，−10% 变差（回放 3.3%→6.7%，外层 5.9%→10%）。
- 按 P(+15%) 排序：外层命中 50.3%，−10% 49.4%。**不得当选点。**

## 代码与产物（诊断，非正式）

- `research/s20_harness/take_profit.py`：+15/−10 配对、沉默、指定上涨
- `research/s20_harness/independent_risk.py`：独立 P(B10) + 滚动校准
- `research/s20_harness/silence.py`：成熟冷却、P(沉默)、P(+15%)、筛选回放
- `research/s20_harness/policy_replay.py`：Dream-RSI 选择层
- `scripts/gate_s20_v4_b10.py`、`scripts/replay_s20_v4_screen.py`
- 写入 `output/experiments/s20_safe_v4/sources/v4-paired-campaign/`（只读冻结排序在 `v4-rsi-campaign`）
- 记录：`docs/research/s20_v4_opt_shipped_policy_20260920.md`、`docs/research/s20_v4_screen_pred_20260920.md`、`docs/research/s20_v4_b10_calibrated_gate_20260920.md`
- 辩论：`wiki/smartgoal/2026-09-20_s20_profit_rsi_explore.md`

## 明确未做

- 没有重训生产 S20，没有替换冻结 multinomial_base 为发运排序。
- 没有正式 H00–H10 验收，没有 H04。
- 没有改生产池或 21:00。

## 下一步（若继续）

仍在 Harness-RSI L1：打 C / 指定 +15%，约束 −10% ≤ π0。不要事后删沉默、不要外层改 8% 或 τ、不要上序列模型除非 L1 证明分数不可分。无合格挑战者则 π0 留任仍是完成态。
