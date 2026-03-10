# StockAgent 系统优化执行方案

> 目标：提升评分对未来上涨确定性的预测准确度，增强LLM参与深度，补齐数据短板
> 原则：每个步骤独立可验证，支持断点续做，后续步骤不依赖前置步骤的"完美"

---

## 执行检查清单

- [x] Step 1: LLM评分融合（P0）
- [x] Step 2: 基本面数据补齐（P0）
- [x] Step 3: 融资融券+北向资金（P1）
- [x] Step 4: Prompt结构化输出（P1）
- [x] Step 5: 回测验证框架（P1）
- [x] Step 6: 市场状态识别（P2）

---

## Step 1: LLM评分融合进最终决策

**现状**：`orchestrator.py:475` 使用 `local_scores`，LLM 评分存在 `ProviderResult.scores` 中但被忽略。

**修改文件**：`src/stockagent_analysis/orchestrator.py`

### 1.1 在最终评分处融合 LLM 评分

在 `orchestrator.py` 的 `model_totals` 计算段（约 line 463-480），修改评分来源：

```python
# 当前逻辑（仅用本地评分）:
score = local_scores.get(a.agent_id, res.score_0_100)

# 改为（融合LLM评分）:
local_s = local_scores.get(a.agent_id, res.score_0_100)
llm_s = provider_results[p].scores.get(a.agent_id)
if llm_s is not None and 0 <= llm_s <= 100:
    score = local_s * 0.6 + llm_s * 0.4   # 本地60% + LLM40%
else:
    score = local_s  # LLM无评分时退回本地
```

### 1.2 新增融合比例配置

在 `configs/project.json` 新增字段：

```json
{
  "score_fusion": {
    "local_weight": 0.6,
    "llm_weight": 0.4
  }
}
```

`orchestrator.py` 从配置读取比例，避免硬编码。

### 1.3 在最终输出中记录 LLM 评分

在 `final_decision.json` 的 detail 段中，每个 agent 增加 `llm_scores` 字段：

```json
{
  "agent_id": "kline_pattern_agent",
  "score_0_100": 50.0,
  "llm_scores": {"kimi": 50, "gemini": 45, "grok": 50, "qwen": 50},
  "fused_score": 48.0
}
```

### 验证

运行一支股票（如 002843），检查：
1. `final_decision.json` 中出现 `llm_scores` 和 `fused_score` 字段
2. 最终分数与之前不同（因融合了LLM评分）
3. 分数仍在合理范围 [0, 100]

---

## Step 2: 基本面数据补齐

**现状**：`_fetch_fundamentals` 只获取 PE/PB/换手率/市值 4个字段。

**修改文件**：`src/stockagent_analysis/data_backend.py`、`src/stockagent_analysis/agents.py`

### 2.1 扩展 `_fetch_fundamentals` 方法

在 Tushare 分支中追加获取：

```python
# 在现有 pro.daily_basic() 之后，追加：
# (1) 财务指标
try:
    _tushare_throttle()
    fina = pro.fina_indicator(ts_code=ts_code, limit=1)
    if not fina.empty:
        row_f = fina.iloc[0]
        data["roe"] = self._safe_float(row_f.get("roe"))                    # ROE(%)
        data["grossprofit_margin"] = self._safe_float(row_f.get("grossprofit_margin"))  # 毛利率
        data["netprofit_margin"] = self._safe_float(row_f.get("netprofit_margin"))      # 净利率
        data["debt_to_assets"] = self._safe_float(row_f.get("debt_to_assets"))          # 资产负债率
        data["revenue_yoy"] = self._safe_float(row_f.get("tr_yoy"))         # 营收同比增速
        data["netprofit_yoy"] = self._safe_float(row_f.get("q_profit_yoy")) # 净利润同比增速
        data["eps"] = self._safe_float(row_f.get("eps"))                    # 每股收益
        data["cfps"] = self._safe_float(row_f.get("cfps"))                  # 每股现金流
except Exception:
    pass

# (2) 十大流通股东变动（最新一期）
try:
    _tushare_throttle()
    holders = pro.top10_floatholders(ts_code=ts_code, limit=10)
    if not holders.empty:
        data["top10_float_holders"] = holders[["holder_name", "hold_amount"]].to_dict("records")[:5]
except Exception:
    pass
```

### 2.2 更新 `_simple_policy` 基本面评分

在 `agents.py` 的 FUNDAMENTAL 维度评分中使用新字段：

```python
# 现有 PE/PB bias 之后追加：
roe = float(fundamentals.get("roe") or 0)
rev_yoy = float(fundamentals.get("revenue_yoy") or 0)
np_yoy = float(fundamentals.get("netprofit_yoy") or 0)
debt = float(fundamentals.get("debt_to_assets") or 0)

# ROE 评分
if roe > 20: roe_bias = 8
elif roe > 10: roe_bias = 4
elif roe > 0: roe_bias = 0
else: roe_bias = -6

# 成长性评分
growth_bias = 0
if rev_yoy > 30 and np_yoy > 30: growth_bias = 10
elif rev_yoy > 15 and np_yoy > 15: growth_bias = 5
elif rev_yoy < -10 or np_yoy < -20: growth_bias = -8

# 负债率
debt_bias = 0
if debt > 80: debt_bias = -6
elif debt > 60: debt_bias = -3
elif debt < 30: debt_bias = 3

base_dim["FUNDAMENTAL"] = 50 + pe_bias + pb_bias + roe_bias + growth_bias + debt_bias + 0.15 * mom
```

### 2.3 更新 `AnalystAgent._build_data_context`

在数据上下文中展示新增基本面字段：

```python
fund = ctx.get("fundamentals", {})
fund_parts = []
if fund.get("roe") is not None:
    fund_parts.append(f"ROE={fund['roe']:.1f}%")
if fund.get("revenue_yoy") is not None:
    fund_parts.append(f"营收增速={fund['revenue_yoy']:.1f}%")
if fund.get("netprofit_yoy") is not None:
    fund_parts.append(f"净利润增速={fund['netprofit_yoy']:.1f}%")
if fund.get("debt_to_assets") is not None:
    fund_parts.append(f"资产负债率={fund['debt_to_assets']:.1f}%")
if fund_parts:
    parts.append("基本面: " + " | ".join(fund_parts))
```

### 验证

运行一支股票，检查：
1. `analysis_context.json` 中 `fundamentals` 包含 roe/revenue_yoy/netprofit_yoy 等字段
2. FUNDAMENTAL 维度评分不再只有 PE/PB 驱动
3. 不报错（Tushare fina_indicator 权限不足时应 graceful fallback）

---

## Step 3: 融资融券 + 北向资金数据

**现状**：`deriv_margin_agent` 配置了需要融资融券数据，但从未获取。

**修改文件**：`src/stockagent_analysis/data_backend.py`、`src/stockagent_analysis/agents.py`

### 3.1 新增 `_fetch_margin_data` 方法

```python
def _fetch_margin_data(self, symbol: str) -> dict[str, Any]:
    """获取融资融券数据（最近20个交易日）。"""
    if not self._tushare_token:
        return {}
    try:
        import tushare as ts
        _tushare_throttle()
        pro = ts.pro_api(timeout=self._tushare_timeout)
        ts_code = self._to_ts_code(symbol)
        margin = pro.margin_detail(ts_code=ts_code, limit=20)
        if margin is None or margin.empty:
            return {}
        latest = margin.iloc[0]
        return {
            "rzye": self._safe_float(latest.get("rzye")),            # 融资余额
            "rzmre": self._safe_float(latest.get("rzmre")),           # 融资买入额
            "rqye": self._safe_float(latest.get("rqye")),             # 融券余额
            "rqmcl": self._safe_float(latest.get("rqmcl")),           # 融券卖出量
            "rzye_change_5d": self._calc_series_change(margin, "rzye", 5),   # 5日融资变化%
            "rzye_change_10d": self._calc_series_change(margin, "rzye", 10),  # 10日融资变化%
            "source": "tushare",
        }
    except Exception:
        return {}
```

### 3.2 新增 `_fetch_hsgt_data` 方法

```python
def _fetch_hsgt_data(self, symbol: str) -> dict[str, Any]:
    """获取北向资金（沪深港通）持股数据。"""
    if not self._tushare_token:
        return {}
    try:
        import tushare as ts
        _tushare_throttle()
        pro = ts.pro_api(timeout=self._tushare_timeout)
        ts_code = self._to_ts_code(symbol)
        hsgt = pro.hk_hold(ts_code=ts_code, limit=20)
        if hsgt is None or hsgt.empty:
            return {}
        latest = hsgt.iloc[0]
        prev = hsgt.iloc[-1] if len(hsgt) > 1 else latest
        vol = self._safe_float(latest.get("vol"))           # 持股量
        ratio = self._safe_float(latest.get("ratio"))       # 持股占比
        prev_vol = self._safe_float(prev.get("vol"))
        change_pct = ((vol - prev_vol) / prev_vol * 100) if prev_vol and prev_vol > 0 else 0
        return {
            "hk_hold_vol": vol,
            "hk_hold_ratio": ratio,
            "hk_hold_change_pct": round(change_pct, 2),
            "source": "tushare",
        }
    except Exception:
        return {}
```

### 3.3 集成到 `_fetch_complete` 并存入 `analysis_context`

在 `_fetch_complete` 中调用新方法，结果存入 features：

```python
features["margin_data"] = self._fetch_margin_data(symbol)
features["hsgt_data"] = self._fetch_hsgt_data(symbol)
```

### 3.4 辅助方法

```python
def _calc_series_change(self, df, col: str, days: int) -> float | None:
    """计算某列过去N日变化百分比。"""
    try:
        if len(df) < days + 1 or col not in df.columns:
            return None
        latest = float(df.iloc[0][col])
        prev = float(df.iloc[min(days, len(df) - 1)][col])
        if prev == 0:
            return None
        return round((latest - prev) / abs(prev) * 100, 2)
    except Exception:
        return None
```

### 3.5 更新 DERIV_MARGIN 维度评分

```python
# 在 _simple_policy 的 DERIV_MARGIN 段
margin = features.get("margin_data", {})
hsgt = features.get("hsgt_data", {})

margin_bias = 0
rzye_5d = float(margin.get("rzye_change_5d") or 0)
if rzye_5d > 5: margin_bias = 6       # 融资余额5日增长>5%: 加杠杆做多
elif rzye_5d < -5: margin_bias = -6    # 融资余额5日下降>5%: 去杠杆

hsgt_bias = 0
hk_chg = float(hsgt.get("hk_hold_change_pct") or 0)
if hk_chg > 3: hsgt_bias = 5          # 北向资金大幅加仓
elif hk_chg < -3: hsgt_bias = -5       # 北向资金大幅减仓

base_dim["DERIV_MARGIN"] = 50 + margin_bias + hsgt_bias + 0.3 * pct - 0.15 * vol
```

### 验证

运行一支融资融券标的（如 000001 平安银行），检查：
1. `analysis_context.json` 出现 `margin_data` 和 `hsgt_data`
2. DERIV_MARGIN 评分因新数据变化
3. Tushare 权限不足时不崩溃（graceful fallback 到空 dict）

---

## Step 4: Prompt 结构化输出 + Few-shot

**现状**：LLM prompt 无示例、无结构化约束，评分解析靠正则匹配最后一个数字。

**修改文件**：`src/stockagent_analysis/llm_client.py`、`src/core/router.py`

### 4.1 改造 `enrich_and_score` prompt

将当前的自由文本 prompt 改为要求 JSON 输出，并加入 few-shot 示例：

```python
prompt = (
    f"你是中国股市{role}分析员。基于以下数据与本地分析结论，给出你的独立研判。\n\n"
    f"股票: {symbol} {name}\n"
    f"本地结论: {base_reason}\n\n"
    f"【数据（供参考，不得编造）】\n{data_context}\n\n"
    f"【评分标准校准】\n"
    f"- 80-100: 强烈看多，你认为未来20日上涨概率>75%\n"
    f"- 60-79: 偏多，上涨概率55-75%\n"
    f"- 40-59: 中性/不确定\n"
    f"- 20-39: 偏空，下跌概率55-75%\n"
    f"- 0-19: 强烈看空，下跌概率>75%\n\n"
    f"【输出示例】\n"
    f'{{"analysis":"MACD金叉配合放量突破20日均线，趋势转强，但RSI接近超买区需警惕短期回调。",'
    f'"score":68,'
    f'"risk":"RSI超买+上方前高压力"}}\n\n'
    f"请仅输出一个JSON对象，包含 analysis(2-3句分析)、score(0-100整数)、risk(主要风险，1句话)："
)
```

### 4.2 改进 `_parse_score_from_response` 解析

在 `core/router.py` 中优先 JSON 解析：

```python
def _parse_score_from_response(text: str, provider_hint: str = "") -> float | None:
    import json, re
    # 1. 优先尝试 JSON
    try:
        # 提取可能的 JSON 块
        json_match = re.search(r'\{[^{}]*"score"\s*:\s*\d+[^{}]*\}', text, re.S)
        if json_match:
            obj = json.loads(json_match.group())
            s = float(obj["score"])
            if 0 <= s <= 100:
                return s
    except Exception:
        pass
    # 2. 回退到现有正则逻辑（保持不变）
    ...
```

### 4.3 提取 risk 字段

`enrich_and_score` 返回值从 `(text, score)` 扩展为 `(text, score, risk)`：

```python
# 解析 JSON 后
analysis = obj.get("analysis", "")
score_val = obj.get("score")
risk = obj.get("risk", "")
return analysis, round(score_val, 2), risk
```

注意：需在 `parallel_runner.py` 调用侧兼容新返回值。存储 risk 到 `result.risks[aid] = risk`。

### 验证

运行一支股票，检查：
1. Agent 日志中 LLM 返回格式为 JSON
2. 评分解析成功率（不应出现 50.0 默认回退）
3. `submissions/*.json` 中出现 `risk` 字段

---

## Step 5: 回测验证框架

**现状**：无任何历史验证机制。

**新增文件**：`src/stockagent_analysis/backtest.py`

### 5.1 信号记录模块

在每次运行结束后，自动保存信号记录到 CSV：

```python
# backtest.py
import csv, os
from datetime import datetime
from pathlib import Path

SIGNAL_DB = Path("output/signal_history.csv")
COLUMNS = [
    "date", "symbol", "name", "final_score", "decision",
    "close_price", "pe_ttm", "momentum_20", "volatility_20",
    # 各维度分数
    "TREND", "TECH", "CAPITAL_FLOW", "FUNDAMENTAL", "KLINE_PATTERN",
    "DIVERGENCE", "SUPPORT_RESISTANCE", "CHANLUN",
    # 后续涨跌（回验时填入）
    "ret_5d", "ret_10d", "ret_20d",
]

def record_signal(run_dir: Path, final_decision: dict, detail: list[dict],
                  snap: dict, features: dict):
    """运行结束后记录信号到 signal_history.csv。"""
    row = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "symbol": snap.get("symbol"),
        "name": snap.get("name"),
        "final_score": final_decision.get("score"),
        "decision": final_decision.get("decision"),
        "close_price": snap.get("close"),
        "pe_ttm": snap.get("pe_ttm"),
        "momentum_20": features.get("momentum_20"),
        "volatility_20": features.get("volatility_20"),
    }
    # 各维度分数
    for d in detail:
        dim = d.get("dim_code", "")
        if dim in COLUMNS:
            row[dim] = d.get("fused_score", d.get("score_0_100"))
    # 后验字段留空
    row["ret_5d"] = ""
    row["ret_10d"] = ""
    row["ret_20d"] = ""

    file_exists = SIGNAL_DB.exists()
    with open(SIGNAL_DB, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if not file_exists:
            w.writeheader()
        w.writerow(row)
```

### 5.2 回验命令

新增 CLI 子命令 `python run.py backfill`：

```python
def backfill_returns():
    """读取 signal_history.csv 中 ret_5d/10d/20d 为空的记录，用实际数据回填。"""
    import pandas as pd

    df = pd.read_csv(SIGNAL_DB)
    empty_mask = df["ret_5d"].isna() | (df["ret_5d"] == "")
    for idx, row in df[empty_mask].iterrows():
        symbol = row["symbol"]
        signal_date = row["date"]
        # 获取 signal_date 之后的实际日线数据
        hist = _fetch_daily_after(symbol, signal_date, days=25)
        if hist is not None and len(hist) >= 5:
            base_close = row["close_price"]
            df.at[idx, "ret_5d"] = round((hist.iloc[4]["close"] / base_close - 1) * 100, 2)
        if hist is not None and len(hist) >= 10:
            df.at[idx, "ret_10d"] = round((hist.iloc[9]["close"] / base_close - 1) * 100, 2)
        if hist is not None and len(hist) >= 20:
            df.at[idx, "ret_20d"] = round((hist.iloc[19]["close"] / base_close - 1) * 100, 2)
    df.to_csv(SIGNAL_DB, index=False, encoding="utf-8")
```

### 5.3 评估报告命令

新增 `python run.py evaluate`：

```python
def evaluate_accuracy():
    """统计信号准确率和各维度IC。"""
    import pandas as pd, numpy as np

    df = pd.read_csv(SIGNAL_DB)
    df = df.dropna(subset=["ret_10d"])

    # 整体胜率
    buy_signals = df[df["decision"] == "buy"]
    win_rate = (buy_signals["ret_10d"] > 0).mean() if len(buy_signals) > 0 else 0

    # 各维度 IC (与 ret_10d 的相关系数)
    dim_cols = ["TREND", "TECH", "CAPITAL_FLOW", "FUNDAMENTAL", "KLINE_PATTERN",
                "DIVERGENCE", "SUPPORT_RESISTANCE", "CHANLUN"]
    ic_report = {}
    for col in dim_cols:
        if col in df.columns:
            valid = df[[col, "ret_10d"]].dropna()
            if len(valid) > 5:
                ic_report[col] = round(valid[col].corr(valid["ret_10d"]), 4)

    print(f"信号总数: {len(df)}")
    print(f"Buy信号胜率(10日): {win_rate:.1%}")
    print(f"各维度IC: {ic_report}")
```

### 5.4 集成到 orchestrator

在 `orchestrator.py` 的 `_run_pipeline` 末尾（PDF生成之后）调用：

```python
from .backtest import record_signal
record_signal(run_dir, final_decision_dict, detail, snap_dict, features)
```

### 验证

1. 运行3-5支股票
2. 检查 `output/signal_history.csv` 生成且包含正确记录
3. 运行 `python run.py backfill` 回填历史涨跌（如果有足够历史数据）

---

## Step 6: 市场状态识别 + 动态阈值

**现状**：buy>=70 / sell<50 阈值固定，不区分牛熊。

**修改文件**：`src/stockagent_analysis/data_backend.py`、`src/stockagent_analysis/orchestrator.py`

### 6.1 新增 `_detect_market_regime` 方法

在 `data_backend.py` 中：

```python
@staticmethod
def _detect_market_regime() -> dict[str, Any]:
    """基于沪深300判断当前市场状态。"""
    try:
        import akshare as ak
        # 获取沪深300最近60个交易日
        idx = ak.stock_zh_index_daily(symbol="sh000300")
        if idx is None or len(idx) < 60:
            return {"regime": "unknown"}
        close = idx["close"].astype(float).tail(60)
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]
        curr = float(close.iloc[-1])
        ret_20d = (curr / float(close.iloc[-20]) - 1) * 100
        vol_20d = float(close.pct_change().tail(20).std() * 100)

        if curr > ma20 > ma60 and ret_20d > 3:
            regime = "bull"
        elif curr < ma20 < ma60 and ret_20d < -3:
            regime = "bear"
        else:
            regime = "range"

        return {
            "regime": regime,
            "index_close": round(curr, 2),
            "index_ma20": round(float(ma20), 2),
            "index_ma60": round(float(ma60), 2),
            "index_ret_20d": round(ret_20d, 2),
            "index_vol_20d": round(vol_20d, 2),
        }
    except Exception:
        return {"regime": "unknown"}
```

### 6.2 动态调整阈值

在 `orchestrator.py` 的最终决策段：

```python
regime = analysis_context.get("features", {}).get("market_regime", {}).get("regime", "unknown")

# 基于市场状态调整阈值
if regime == "bull":
    buy_th = 65     # 牛市降低买入门槛
    sell_th = 45
elif regime == "bear":
    buy_th = 78     # 熊市提高买入门槛
    sell_th = 55
else:
    buy_th = config.get("decision_threshold_buy", 70)
    sell_th = config.get("decision_threshold_sell", 50)
```

### 6.3 集成到数据采集

在 `_fetch_complete` 或 `_compute_features` 中调用：

```python
features["market_regime"] = self._detect_market_regime()
```

### 验证

1. 检查 `analysis_context.json` 中出现 `market_regime` 字段
2. 不同市场状态下阈值正确调整
3. 最终决策日志打印当前市场状态和使用的阈值

---

## 断点续做说明

每个 Step 完全独立，可单独完成和验证：

| Step | 依赖 | 可独立执行 |
|------|------|-----------|
| Step 1 (LLM评分融合) | 无 | 是 |
| Step 2 (基本面补齐) | 无 | 是 |
| Step 3 (融资融券) | 无 | 是 |
| Step 4 (Prompt结构化) | 无 | 是 |
| Step 5 (回测框架) | Step 1 最佳（需要 fused_score） | 是（无 fused_score 时用 score_0_100） |
| Step 6 (市场状态) | 无 | 是 |

**续做方式**：检查上方清单中的 `[x]` 标记，从第一个未完成的 Step 继续。每个 Step 完成后运行验证项，通过后标记为完成。

---

## 不做的事（避免过度工程）

1. **不做自动权重优化** — 需要大量历史数据，Step 5 的信号积累是前提
2. **不做多模型专长分工** — 当前四模型均能胜任所有维度，分工收益不大
3. **不做行业相对估值** — 需要行业数据库，维护成本高
4. **不做实时盯盘** — 系统定位是"分析研判"，非"交易执行"
5. **不改 agent 数量/结构** — 29个 agent 已覆盖主要维度，避免架构变动风险
