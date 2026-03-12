# StockAgent 系统优化执行方案（第二期）

> 目标：补齐操作可执行性短板 — 追高过滤、结构化点位、分场景建议、回测验证、策略框架、筹码数据
> 原则：每个步骤独立可验证，支持断点续做，后续步骤不依赖前置步骤的"完美"

---

## 执行检查清单

- [ ] Step 1: 乖离率追高过滤（P1）
- [ ] Step 2: 结构化狙击点位（P2）
- [ ] Step 3: 空仓/持仓分别建议（P3）
- [ ] Step 4: 回测评估引擎（P4）
- [ ] Step 5: 市场策略框架增强（P5）
- [ ] Step 6: 筹码结构分析（P6）

---

## Step 1: 乖离率追高过滤

**现状**：`agents.py` `_simple_policy()` 的 TREND 公式为 `50 + 1.0*mom + 0.8*trend + 0.3*pct + 0.12*dd`，已在 `kline_indicators.day.ma_system.ma5.pct_above` 计算了乖离率，但从未用于过滤或惩罚。高乖离率时系统仍可能给出买入建议，导致追高。

**修改文件**：`src/stockagent_analysis/agents.py`、`src/stockagent_analysis/orchestrator.py`、`configs/project.json`

### 1.1 修改 TREND 维度公式（agents.py）

在 `_simple_policy()` 的 TREND 计算段（约 line 319-325），读取乖离率并加入惩罚：

```python
# 读取MA5乖离率
ma_sys = features.get("kline_indicators", {}).get("day", {}).get("ma_system", {})
bias_ma5 = float(ma_sys.get("ma5", {}).get("pct_above", 0))

# 乖离率惩罚
bias_penalty = 0
if abs(bias_ma5) > 8:
    bias_penalty = -25  # 极端乖离，大幅惩罚
elif abs(bias_ma5) > 5:
    bias_penalty = -15  # 中度乖离，中幅惩罚
elif abs(bias_ma5) > 3:
    bias_penalty = -5   # 轻度乖离，小幅惩罚

base_dim["TREND"] = 50 + 1.0*mom + 0.8*trend + 0.3*pct + 0.12*dd + bias_penalty
```

同时对正乖离极端值硬封顶：

```python
# 极端追高硬封顶（MA5乖离>8%时，TREND分数不超过35）
if bias_ma5 > 8:
    base_dim["TREND"] = min(base_dim["TREND"], 35)
```

### 1.2 orchestrator 后置过滤（orchestrator.py）

在最终决策段（约 line 531-554），`final_score` 计算完毕后、决策判定前，增加后置过滤：

```python
# 读取乖离率
bias_pct = analysis_context.get("features", {}).get("kline_indicators", {}) \
    .get("day", {}).get("ma_system", {}).get("ma5", {}).get("pct_above", 0)
bias_threshold = config.get("bias_threshold_pct", 5.0)

# 后置过滤：超阈值降一级决策 + 追加警告
bias_warning = ""
if bias_pct > bias_threshold:
    if decision_level in ("strong_buy", "weak_buy"):
        decision_level = "hold"  # 降一级
        decision = "hold"
        bias_warning = f"⚠️ 乖离率警告: MA5乖离{bias_pct:.1f}%>{bias_threshold}%，建议等待回踩后介入"
    elif decision_level == "hold":
        bias_warning = f"⚠️ 乖离率偏高({bias_pct:.1f}%)，谨慎追高"
```

将 `bias_warning` 写入 `final_decision.json` 的 `warnings` 字段。

### 1.3 project.json 配置

新增乖离率阈值配置：

```json
{
  "bias_filter": {
    "enabled": true,
    "bias_threshold_pct": 5.0,
    "hard_cap_pct": 8.0,
    "hard_cap_max_score": 35
  }
}
```

### 验证

运行一支近期大涨股票（MA5乖离率>5%），检查：
1. TREND 维度分数因乖离率惩罚降低
2. 若原本为 buy，决策降级为 hold + 出现 bias_warning
3. `final_decision.json` 中出现 `warnings` 字段
4. MA5乖离率正常（<3%）的股票不受影响

---

## Step 2: 结构化狙击点位（买入/止损/目标价）

**现状**：`llm_client.py` 的 `generate_scenario_and_position()`（line 280-318）输出自由文本场景和仓位建议，无结构化价位数据。投资者无法直接获取买入/止损/目标价。

**修改文件**：`src/stockagent_analysis/llm_client.py`、`src/stockagent_analysis/orchestrator.py`、`src/stockagent_analysis/report_pdf.py`、`src/stockagent_analysis/backtest.py`

### 2.1 修改 generate_scenario_and_position prompt（llm_client.py）

在 `generate_scenario_and_position()` 中，将自由文本 prompt 改为要求返回包含结构化点位的 JSON：

```python
prompt = (
    f"基于以下分析结论和数据，为 {symbol} {name} 生成操作建议。\n\n"
    f"当前价: {current_price}\n"
    f"分析结论: {analysis_summary}\n"
    f"技术指标: {tech_context}\n\n"
    f"请仅输出一个JSON对象，包含以下字段：\n"
    f'{{"scenarios": {{"optimistic": {{"probability": 35, "target": 价格, "reason": "一句话"}}, '
    f'"neutral": {{"probability": 45, "target": 价格, "reason": "一句话"}}, '
    f'"pessimistic": {{"probability": 20, "target": 价格, "reason": "一句话"}}}}, '
    f'"sniper_points": {{"ideal_buy": 首选买入价, "secondary_buy": 次选买入价, '
    f'"stop_loss": 止损价, "take_profit_1": 第一目标价, "take_profit_2": 第二目标价}}, '
    f'"position_strategy": "分批建仓策略文字描述"}}\n'
)
```

### 2.2 解析 sniper_points（llm_client.py）

新增解析逻辑，返回值从 `(scenario_text, position_text)` 扩展为 `(scenarios, sniper_points, position_strategy)`：

```python
def generate_scenario_and_position(router, symbol, name, current_price,
                                    analysis_summary, tech_context) -> tuple:
    ...
    try:
        obj = json.loads(response_text)
        scenarios = obj.get("scenarios", {})
        sniper_points = obj.get("sniper_points", {})
        position_strategy = obj.get("position_strategy", "")
        return scenarios, sniper_points, position_strategy
    except json.JSONDecodeError:
        # 回退到原有自由文本解析
        return _parse_freetext_scenario(response_text)
```

### 2.3 orchestrator 输出 sniper_points（orchestrator.py）

在 `final_decision.json` 中新增字段：

```python
final_decision_dict["sniper_points"] = sniper_points
final_decision_dict["scenarios"] = scenarios
```

### 2.4 report_pdf 新增狙击点位表格（report_pdf.py）

在 `build_investor_pdf()` 的 scenario 段后增加 `_add_sniper_points_table()`：

```python
def _add_sniper_points_table(self, story, sniper_points, current_price):
    """添加狙击点位表格。"""
    data = [["点位类型", "价格", "距当前价", "说明"]]
    fields = [
        ("ideal_buy", "首选买入", "支撑位附近低吸"),
        ("secondary_buy", "次选买入", "二次确认位"),
        ("stop_loss", "止损价", "跌破即离场"),
        ("take_profit_1", "目标价1", "第一止盈目标"),
        ("take_profit_2", "目标价2", "第二止盈目标"),
    ]
    for key, label, desc in fields:
        price = sniper_points.get(key)
        if price:
            diff_pct = (price - current_price) / current_price * 100
            data.append([label, f"{price:.2f}", f"{diff_pct:+.1f}%", desc])
    # 构建 Table ...
```

### 2.5 backtest 记录点位（backtest.py）

在 `COLUMNS` 中新增列：

```python
COLUMNS = [
    ...existing columns...,
    "ideal_buy", "stop_loss", "take_profit_1",
    "direction_expected",  # "up" / "down" / "neutral"
]
```

`record_signal()` 中记录：

```python
sp = final_decision.get("sniper_points", {})
row["ideal_buy"] = sp.get("ideal_buy")
row["stop_loss"] = sp.get("stop_loss")
row["take_profit_1"] = sp.get("take_profit_1")
row["direction_expected"] = "up" if final_decision.get("decision") == "buy" else \
                            "down" if final_decision.get("decision") == "sell" else "neutral"
```

### 验证

运行一支股票，检查：
1. `final_decision.json` 中出现 `sniper_points` 和 `scenarios` 字段
2. 价位合理（ideal_buy < current_price < take_profit）
3. PDF 报告中有狙击点位表格
4. `signal_history.csv` 新增 ideal_buy/stop_loss/take_profit_1 列

---

## Step 3: 空仓/持仓分别建议

**现状**：系统输出单一决策（如 65 分 "hold"），但对空仓者意味着"不买"，对持仓者意味着"继续持有"，含义完全不同。

**修改文件**：`src/stockagent_analysis/llm_client.py`、`src/stockagent_analysis/report_pdf.py`

### 3.1 修改 LLM prompt（llm_client.py）

与 Step 2 共用同一 LLM 调用，在 `generate_scenario_and_position()` 的 JSON 输出中追加字段：

```python
# 在 prompt JSON 模板中追加:
f'"position_advice": {{'
f'"no_position": "空仓者操作建议（1-2句）", '
f'"has_position": "持仓者操作建议（1-2句）", '
f'"position_ratio": "建议仓位比例(0-100%)"}}'
```

完整返回值扩展为：`(scenarios, sniper_points, position_strategy, position_advice)`

### 3.2 orchestrator 输出（orchestrator.py）

```python
final_decision_dict["position_advice"] = position_advice
# position_advice = {
#   "no_position": "建议分3批在18.5-19.2元区间低吸，首批30%",
#   "has_position": "持有观望，跌破18元止损",
#   "position_ratio": "50%"
# }
```

### 3.3 report_pdf 双列展示（report_pdf.py）

在 `_add_scenario_table()` 后新增 `_add_position_advice_table()`：

```python
def _add_position_advice_table(self, story, position_advice):
    """添加空仓/持仓分别建议表。"""
    data = [
        ["场景", "操作建议"],
        ["空仓（未持有）", position_advice.get("no_position", "暂无建议")],
        ["持仓（已持有）", position_advice.get("has_position", "暂无建议")],
        ["建议仓位", position_advice.get("position_ratio", "N/A")],
    ]
    # 构建 Table with 两列布局 ...
```

### 验证

运行一支股票，检查：
1. `final_decision.json` 中出现 `position_advice` 字段，含 `no_position` 和 `has_position`
2. 两种建议逻辑一致但角度不同
3. PDF 报告中有空仓/持仓分别建议表格
4. 对于 strong_buy 股票，空仓建议为"买入"，持仓建议为"加仓"（语义合理）

---

## Step 4: 回测评估引擎

**现状**：`backtest.py`（168行）仅有信号记录（`record_signal`）、回填（`backfill_returns`）、基础IC统计（`evaluate_accuracy`），缺少方向准确率、止盈止损命中率、模拟收益率等核心回测指标。

**修改文件**：`src/stockagent_analysis/backtest.py`、`run.py`（或 `main.py`）

### 4.1 新增方向判断函数（backtest.py）

```python
def infer_direction(decision: str, score: float) -> str:
    """从决策推断预期方向。"""
    if decision in ("buy", "strong_buy") or score >= 65:
        return "up"
    elif decision in ("sell", "strong_sell") or score < 40:
        return "down"
    return "neutral"
```

### 4.2 新增单信号评估（backtest.py）

```python
def evaluate_single(row: dict) -> dict:
    """评估单条信号的准确性。"""
    direction = row.get("direction_expected", infer_direction(row["decision"], row["final_score"]))
    ret_10d = float(row.get("ret_10d", 0) or 0)
    stop_loss = float(row.get("stop_loss", 0) or 0)
    take_profit = float(row.get("take_profit_1", 0) or 0)
    close = float(row.get("close_price", 0))

    # 方向准确率
    direction_correct = (direction == "up" and ret_10d > 0) or \
                        (direction == "down" and ret_10d < 0) or \
                        (direction == "neutral" and abs(ret_10d) < 2)

    # 止损/止盈命中
    stop_hit = stop_loss > 0 and close * (1 + ret_10d/100) <= stop_loss
    tp_hit = take_profit > 0 and close * (1 + ret_10d/100) >= take_profit

    # 模拟收益（假设按建议操作）
    if direction == "up":
        sim_return = ret_10d  # 做多
    elif direction == "down":
        sim_return = -ret_10d  # 做空/空仓
    else:
        sim_return = 0  # 观望不操作

    return {
        "direction_correct": direction_correct,
        "stop_hit": stop_hit,
        "tp_hit": tp_hit,
        "simulated_return_pct": round(sim_return, 2),
        "outcome": "win" if sim_return > 0 else "loss" if sim_return < 0 else "flat",
    }
```

### 4.3 新增汇总统计（backtest.py）

```python
def compute_summary(df: pd.DataFrame) -> dict:
    """计算整体回测统计。"""
    total = len(df)
    if total == 0:
        return {"error": "无可评估信号"}

    evaluated = df.dropna(subset=["ret_10d"])
    n = len(evaluated)
    if n == 0:
        return {"total_signals": total, "evaluated": 0, "note": "尚无回填数据"}

    results = [evaluate_single(row) for _, row in evaluated.iterrows()]

    win_count = sum(1 for r in results if r["outcome"] == "win")
    direction_correct = sum(1 for r in results if r["direction_correct"])
    stop_hits = sum(1 for r in results if r["stop_hit"])
    tp_hits = sum(1 for r in results if r["tp_hit"])
    avg_return = sum(r["simulated_return_pct"] for r in results) / n

    # 分档统计
    buy_signals = evaluated[evaluated["decision"].isin(["buy", "strong_buy"])]
    sell_signals = evaluated[evaluated["decision"].isin(["sell", "strong_sell"])]

    return {
        "total_signals": total,
        "evaluated": n,
        "win_rate": round(win_count / n * 100, 1),
        "direction_accuracy": round(direction_correct / n * 100, 1),
        "stop_loss_hit_rate": round(stop_hits / n * 100, 1) if stop_hits else 0,
        "take_profit_hit_rate": round(tp_hits / n * 100, 1) if tp_hits else 0,
        "avg_simulated_return": round(avg_return, 2),
        "buy_count": len(buy_signals),
        "buy_win_rate": round((buy_signals["ret_10d"] > 0).mean() * 100, 1) if len(buy_signals) > 0 else 0,
        "sell_count": len(sell_signals),
        "sell_win_rate": round((sell_signals["ret_10d"] < 0).mean() * 100, 1) if len(sell_signals) > 0 else 0,
    }
```

### 4.4 新增 CLI 子命令（main.py）

在 `main.py` 的 argparse 中新增 `backtest` 子命令：

```python
sub_bt = subparsers.add_parser("backtest", help="回测评估历史信号")
sub_bt.add_argument("--backfill", action="store_true", help="先回填实际涨跌数据")
sub_bt.add_argument("--days", type=int, default=10, help="评估周期（默认10日）")
sub_bt.set_defaults(func=cmd_backtest)

def cmd_backtest(args):
    from stockagent_analysis.backtest import backfill_returns, compute_summary
    import pandas as pd
    if args.backfill:
        print("正在回填历史涨跌数据...")
        backfill_returns()
    df = pd.read_csv("output/signal_history.csv")
    summary = compute_summary(df)
    # 格式化输出统计表
    for k, v in summary.items():
        print(f"  {k}: {v}")
```

### 验证

1. 积累 5+ 条信号记录后，运行 `python run.py backtest --backfill`
2. 检查输出包含：方向准确率、胜率、模拟平均收益
3. 检查 `signal_history.csv` 中 `ret_5d/10d/20d` 被正确回填
4. 无 Traceback

---

## Step 5: 市场策略框架增强

**现状**：`data_backend.py` 的 `_detect_market_regime()`（line 1201-1231）仅输出 3 态（bull/bear/range），`orchestrator.py` 仅据此调整阈值。缺乏细粒度策略映射（如进攻/平衡/防守阶段的仓位/行业配置建议）。

**修改文件**：新建 `src/stockagent_analysis/market_strategy.py`、修改 `src/stockagent_analysis/data_backend.py`、`src/stockagent_analysis/agents.py`

### 5.1 新建 market_strategy.py

```python
"""A股市场策略框架 — 三阶段策略映射。"""

from dataclasses import dataclass

@dataclass
class MarketStrategy:
    """市场策略建议。"""
    phase: str          # "offensive" / "balanced" / "defensive"
    phase_cn: str       # "进攻" / "平衡" / "防守"
    position_cap: float # 建议最大仓位 (0.0-1.0)
    sector_bias: str    # 行业偏好建议
    risk_note: str      # 风险提示

# A股三阶段策略映射
STRATEGY_MAP = {
    "offensive": MarketStrategy(
        phase="offensive", phase_cn="进攻",
        position_cap=0.9,
        sector_bias="偏好成长股、科技、新能源，可适当追涨强势板块",
        risk_note="注意阶段性顶部信号，设好止盈线",
    ),
    "balanced": MarketStrategy(
        phase="balanced", phase_cn="平衡",
        position_cap=0.6,
        sector_bias="均衡配置，兼顾价值与成长，关注低位补涨板块",
        risk_note="控制仓位，避免单一板块过度集中",
    ),
    "defensive": MarketStrategy(
        phase="defensive", phase_cn="防守",
        position_cap=0.3,
        sector_bias="偏好高股息、消费、公用事业等防御板块",
        risk_note="严格止损，轻仓观望为主",
    ),
}

def determine_strategy(regime: dict) -> MarketStrategy:
    """基于市场状态判定策略阶段。"""
    regime_name = regime.get("regime", "unknown")
    ret_20d = float(regime.get("index_ret_20d", 0))
    vol_20d = float(regime.get("index_vol_20d", 2))

    # 进攻条件: 牛市 + 波动率可控
    if regime_name == "bull" and vol_20d < 2.5:
        return STRATEGY_MAP["offensive"]
    # 防守条件: 熊市 或 高波动
    elif regime_name == "bear" or vol_20d > 3.5:
        return STRATEGY_MAP["defensive"]
    # 默认平衡
    else:
        return STRATEGY_MAP["balanced"]

def strategy_to_dict(strategy: MarketStrategy) -> dict:
    """序列化为JSON可存储格式。"""
    return {
        "phase": strategy.phase,
        "phase_cn": strategy.phase_cn,
        "position_cap": strategy.position_cap,
        "sector_bias": strategy.sector_bias,
        "risk_note": strategy.risk_note,
    }
```

### 5.2 集成到 data_backend.py

在 `_compute_features()` 中，`market_regime` 计算完毕后，追加策略判定：

```python
from .market_strategy import determine_strategy, strategy_to_dict

# 在 _compute_features() 中
regime = self._detect_market_regime()
features["market_regime"] = regime
features["market_strategy"] = strategy_to_dict(determine_strategy(regime))
```

### 5.3 注入 Agent 数据上下文（agents.py）

在 `_build_data_context()` 中，追加策略信息供 LLM 参考：

```python
strategy = ctx.get("features", {}).get("market_strategy", {})
if strategy:
    parts.append(
        f"市场策略: {strategy.get('phase_cn', '未知')}阶段 | "
        f"建议最大仓位{strategy.get('position_cap', 0.6)*100:.0f}% | "
        f"{strategy.get('sector_bias', '')}"
    )
```

### 验证

运行一支股票，检查：
1. `analysis_context.json` 中出现 `market_strategy` 字段
2. 策略阶段与 `market_regime` 逻辑一致（bull→offensive, bear→defensive, range→balanced）
3. Agent LLM 上下文中出现市场策略信息
4. 不同市场状态下策略建议不同

---

## Step 6: 筹码结构分析

**现状**：系统完全没有筹码分布数据（集中度、获利比例、平均成本）。对于判断上方套牢盘压力、底部筹码支撑等缺乏数据支持。

**修改文件**：`src/stockagent_analysis/data_backend.py`、`src/stockagent_analysis/agents.py`、`configs/project.json`

### 6.1 新增 _fetch_chip_distribution（data_backend.py）

基于日线量价数据模拟筹码分布（无需额外数据源）：

```python
def _fetch_chip_distribution(self, hist_df: pd.DataFrame, current_price: float) -> dict:
    """基于日线量价数据估算筹码分布。

    原理：以成交量为权重，按价格区间统计筹码堆积情况。
    近期成交量权重更高（衰减因子）。
    """
    if hist_df is None or len(hist_df) < 60:
        return {}

    df = hist_df.tail(120).copy()  # 取最近120个交易日

    # 价格区间划分（当前价±30%范围，分20档）
    price_min = current_price * 0.7
    price_max = current_price * 1.3
    bins = np.linspace(price_min, price_max, 21)

    # 按成交量加权统计各价格区间筹码量（近期权重更大）
    chip_dist = np.zeros(20)
    for i, (_, row) in enumerate(df.iterrows()):
        decay = 0.95 ** (len(df) - 1 - i)  # 时间衰减
        avg_price = (row["high"] + row["low"] + row["close"]) / 3
        vol = row["volume"] * decay
        bin_idx = np.searchsorted(bins, avg_price) - 1
        if 0 <= bin_idx < 20:
            chip_dist[bin_idx] += vol

    total_chips = chip_dist.sum()
    if total_chips == 0:
        return {}

    chip_pct = chip_dist / total_chips * 100  # 各区间占比

    # 获利比例：当前价以下筹码占比
    current_bin = np.searchsorted(bins, current_price) - 1
    profit_ratio = chip_pct[:max(0, current_bin + 1)].sum()

    # 筹码集中度：前3大区间占比
    top3_bins = np.argsort(chip_pct)[-3:]
    concentration = chip_pct[top3_bins].sum()

    # 平均成本
    bin_centers = (bins[:-1] + bins[1:]) / 2
    avg_cost = np.average(bin_centers, weights=chip_dist) if total_chips > 0 else current_price

    # 上方套牢盘：当前价以上筹码占比
    trapped_ratio = 100 - profit_ratio

    # 筹码健康度评分 (0-100)
    # 获利比例高 + 集中度高 + 套牢少 = 健康
    health = min(100, profit_ratio * 0.4 + concentration * 0.3 + (100 - trapped_ratio) * 0.3)

    return {
        "profit_ratio": round(profit_ratio, 1),       # 获利比例%
        "trapped_ratio": round(trapped_ratio, 1),      # 套牢比例%
        "concentration": round(concentration, 1),      # 筹码集中度%
        "avg_cost": round(float(avg_cost), 2),         # 平均成本
        "health_score": round(health, 1),              # 健康度 0-100
        "current_vs_cost": round((current_price / avg_cost - 1) * 100, 1),  # 当前价vs平均成本%
    }
```

### 6.2 集成到 _compute_features（data_backend.py）

```python
# 在 _compute_features() 中，日线数据可用时
if config.get("enable_chip_distribution", True):
    features["chip_distribution"] = self._fetch_chip_distribution(hist_day, current_price)
```

### 6.3 新增筹码评分公式（agents.py）

在 `_simple_policy()` 中，为现有 FUNDAMENTAL 维度追加筹码因子（不新建 Agent，避免架构变动）：

```python
# 在 FUNDAMENTAL 计算段之后
chip = features.get("chip_distribution", {})
chip_bias = 0
if chip:
    profit_r = float(chip.get("profit_ratio", 50))
    trapped_r = float(chip.get("trapped_ratio", 50))
    health = float(chip.get("health_score", 50))

    # 获利比例高（筹码在手）→ 看多
    if profit_r > 80: chip_bias += 6
    elif profit_r > 60: chip_bias += 3
    elif profit_r < 30: chip_bias -= 6  # 大量套牢盘

    # 上方套牢重 → 压力大
    if trapped_r > 60: chip_bias -= 5

    # 筹码健康度
    chip_bias += (health - 50) * 0.08  # ±4 range

base_dim["FUNDAMENTAL"] += chip_bias  # 叠加到 FUNDAMENTAL
```

### 6.4 注入 LLM 上下文（agents.py）

在 `_build_data_context()` 中追加：

```python
chip = ctx.get("features", {}).get("chip_distribution", {})
if chip:
    parts.append(
        f"筹码分布: 获利{chip.get('profit_ratio', 'N/A')}% | "
        f"套牢{chip.get('trapped_ratio', 'N/A')}% | "
        f"集中度{chip.get('concentration', 'N/A')}% | "
        f"平均成本{chip.get('avg_cost', 'N/A')} | "
        f"健康度{chip.get('health_score', 'N/A')}"
    )
```

### 6.5 project.json 配置开关

```json
{
  "enable_chip_distribution": true
}
```

### 验证

运行一支股票，检查：
1. `analysis_context.json` 中出现 `chip_distribution` 字段
2. 获利比例 + 套牢比例 ≈ 100%
3. FUNDAMENTAL 评分因筹码数据变化（对比无筹码时）
4. Agent LLM 上下文中出现筹码分布信息
5. `enable_chip_distribution: false` 时筹码功能关闭

---

## 断点续做说明

每个 Step 完全独立，可单独完成和验证：

| Step | 依赖 | 可独立执行 |
|------|------|-----------|
| Step 1 (乖离率过滤) | 无 | 是 |
| Step 2 (狙击点位) | 无 | 是 |
| Step 3 (双建议) | Step 2 最佳（共用prompt） | 是（可单独改prompt） |
| Step 4 (回测引擎) | Step 2 最佳（需要sniper_points） | 是（无点位时跳过命中率） |
| Step 5 (市场策略) | 无 | 是 |
| Step 6 (筹码分析) | 无 | 是 |

**续做方式**：检查上方清单中的 `[x]` 标记，从第一个未完成的 Step 继续。每个 Step 完成后运行验证项，通过后标记为完成。

---

## 不做的事（避免过度工程）

1. **不做实时筹码数据接口** — 用日线量价模拟即可，精度足够
2. **不做自动化回测调仓** — 系统定位是"分析研判"，回测仅验证准确性
3. **不做多市场策略** — 仅A股三阶段框架，不扩展到港股/美股
4. **不新建 Agent** — 筹码数据叠加到 FUNDAMENTAL，避免29→30 agent的配置复杂度
5. **不做复杂止损策略** — 狙击点位由LLM给出固定价位，不做动态追踪止损
