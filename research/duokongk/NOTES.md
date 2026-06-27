# duokongK 启动窗口验证 — 移植规范 + 实验设计

起因 (2026-06-13): 用户提供 TradingView Pine `多空K线` 脚本, 提议把 **duokongK 状态机当作独立标注维度**,
定义"底部确认翻多窗口"作择时/onset 标注。本实验**廉价全量验证它是否在 SQUAT 之上多了信息, 还是换皮**。
生产线 V12.31 冻结; 这是描述统计+残差消融, 非建 sleeve。

## 1. duokongK 状态机移植规范 (faithful port, 全因果)

参数 `klen=30`。逐 (ts_code) 按 trade_date 升序, 维护持久状态 `dk_dir∈{0,+1,-1}`, `slHigh`, `slLow`。
每根 K **按此顺序** (与 Pine 一致, 避免单根内乱序):

```
hhk = highest(high, 30)[1]   # 前30根(不含当根)最高
llk = lowest(low, 30)[1]     # 前30根(不含当根)最低

# (a) 首次突破平台
if close > hhk and dk_dir != +1:  dk_dir=+1; slLow=low;  slHigh=0
if close < llk and dk_dir != -1:  dk_dir=-1; slHigh=high; slLow=0
# (b) 突破失败翻转 (用上一根遗留的 slLow/slHigh, 即先判后更新)
if dk_dir==+1 and slLow>0 and close < slLow:  dk_dir=-1; slHigh=high; slLow=0
if dk_dir==-1 and slHigh>0 and close > slHigh: dk_dir=+1; slLow=low;  slHigh=0
# (c) 移动结构止损 (跟极值)
if slHigh>0 and high>slHigh: slHigh=high
if slLow>0  and low<slLow:   slLow=low
```

**全因果**: hhk/llk 用 `[1]`, 状态只依赖历史。`dk_flip = (dk_dir != dk_dir[1])`。
落 `dk_dir, dk_flip, bars_since_flip, dist_to_slLow/slHigh` 每 (ts_code, trade_date)。

## 2. 用户窗口定义 (预注册, 标注层允许用未来确认)

- **空腿局部低点 (pivot low)**: 在 `dk_dir=-1` 段, `low[t]` 是 `[t-5, t+5]` 内最小 (右侧 5 根不破新低 → t+5 确认)。
- **空翻多变色窗口**: pivot low 之后 `dk_dir` 由 −1 翻 +1 且持续 **≥3 根** +1。
- **窗口区域** = [pivot_low_idx ... flip+3]; 标 `dk_window=1`。这区域"含一个或多个启动子"。
- ⚠ pivot 需右侧 5 根确认 → 作 **label/标注合法** (未来数据); 作**实时择时**入场迟到 ~5+3 根 (文档明示)。

## 3. 全量验证 (3 问, 预期: 大面积重叠 SQUAT, 残差≈0 → REJECT_reskin)

1. **画像**: dk_window 后 20/40d 前向: 大涨(≥+15%) / 横盘(|ret|<5%) / 失败(<−5%) 占比 + max_dd; 分 regime。
2. **重叠**: dk_window onset 与 SLV-001 SQUAT 桶 launch / V12.31 选股 的重叠率 (测换皮)。
3. **残差增量 (gate)**: 因果特征 `dk_state/dk_flip_recency/反转确认强度` 前向 IC, 扣 `[past_r5 + pyr_velocity 多窗 + ADX + RSI]` 残差 IC。
   **NEW_INFO = 残差 |IC|≥0.01 且 |t|≥2 且 重叠后残差窗口前向显著兑现; 否则 REJECT_reskin / REJECT_no_residual_pay。**

仅 NEW_INFO 才进后续 walk-forward blend (Phase 2, 本实验不做)。中间画像≠落地 (R03)。
