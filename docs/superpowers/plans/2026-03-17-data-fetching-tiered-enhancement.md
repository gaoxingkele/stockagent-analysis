# 数据获取梯次化增强 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 增强 data_backend.py 的数据获取韧性：5级K线降级、Tushare PE/PB修正层、TDX感知信号量、盘中虚拟K线、腾讯K线直调

**Architecture:** 在现有 TDX→AKShare→Tushare 三级链基础上，插入腾讯K线直调(P3.5)作为AKShare东财接口挂掉时的独立备选；Snapshot增加Tushare修正层(先拿数据再修正PE/PB)；K线批量抓取改为TDX感知的信号量模型；盘中交易时段自动构造虚拟日K线。

**Tech Stack:** Python, akshare, tushare, mootdx, requests, threading.Semaphore, pandas

**数据源优先级（改进后）:**
```
K线: TDX(P1) → AKShare东财(P2) → Tushare(P3) → 腾讯K线直调(P3.5)
快照: TDX → AKShare → Tushare → 最后用Tushare修正PE/PB
```

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/stockagent_analysis/data_backend.py` | Modify | 所有5项改进的主文件 |

所有改进集中在 data_backend.py 一个文件，无需新建文件。

---

### Task 1: 腾讯K线直调 — P3.5降级源

**Files:**
- Modify: `src/stockagent_analysis/data_backend.py:384-438` (`_fetch_multi_timeframe_klines`)
- Modify: `src/stockagent_analysis/data_backend.py` (新增 `_fetch_kline_tencent` 方法)

**说明:** 当AKShare(东财push2)和Tushare都失败时，用腾讯K线 `web.ifzq.gtimg.cn` 作为独立数据源。仅支持日线。

- [ ] **Step 1: 新增 `_fetch_kline_tencent` 方法**

在 `_fetch_kline_tushare` 方法之后（约 line 821），插入新方法：

```python
def _fetch_kline_tencent(self, symbol: str, timeframe: str, limit: int):
    """腾讯K线直调（web.ifzq.gtimg.cn），绕过akshare封装。仅支持日线。"""
    import json as _json
    import pandas as pd
    import requests as _req

    if timeframe != "day":
        raise RuntimeError("tencent_kline_only_daily")

    market_code = f"sh{symbol}" if symbol.startswith("6") else f"sz{symbol}"
    start_date = (datetime.now() - timedelta(days=limit * 2)).strftime("%Y-%m-%d")
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    params = {
        "param": f"{market_code},day,{start_date},,{limit},qfq",
        "_var": "kline_dayqfq",
        "r": str(time.time()),
    }
    resp = _req.get(url, params=params, timeout=10)
    resp.encoding = "utf-8"
    text = resp.text
    if "=" in text:
        text = text.split("=", 1)[1]
    data = _json.loads(text)

    kdata = data.get("data", {}).get(market_code, {})
    bars = kdata.get("qfqday") or kdata.get("day") or []
    if not bars:
        raise RuntimeError("tencent_kline_empty")

    rows = []
    for bar in bars:
        if len(bar) < 6:
            continue
        rows.append({
            "ts": bar[0],
            "open": float(bar[1]),
            "close": float(bar[2]),
            "high": float(bar[3]),
            "low": float(bar[4]),
            "volume": float(bar[5]),
            "amount": 0.0,
            "pct_chg": 0.0,
        })
    if not rows:
        raise RuntimeError("tencent_kline_no_valid_bars")

    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["pct_chg"] = df["close"].pct_change() * 100
    df["pct_chg"] = df["pct_chg"].fillna(0.0)
    df = df.dropna(subset=["close"]).sort_values("ts").reset_index(drop=True)
    if len(df) < 5:
        raise RuntimeError("tencent_kline_insufficient")
    return df.tail(limit)
```

- [ ] **Step 2: 修改 `_fetch_multi_timeframe_klines` 加入腾讯作为P3.5**

修改 `data_backend.py:393`，将 `ordered_sources` 改为：

```python
ordered_sources = ["tdx", "akshare", "tushare", "tencent"]
```

修改 `data_backend.py:410-417` 的 source dispatch，增加 tencent 分支：

```python
if source == "tdx":
    data = self._fetch_kline_tdx(symbol, tf, limit)
elif source == "akshare":
    data = self._fetch_kline_akshare(symbol, tf, limit)
elif source == "tushare":
    data = self._fetch_kline_tushare(symbol, tf, limit)
elif source == "tencent":
    data = self._fetch_kline_tencent(symbol, tf, limit)
else:
    raise ValueError(f"unsupported source: {source}")
```

- [ ] **Step 3: 验证**

```bash
python -c "
from src.stockagent_analysis.data_backend import DataBackend
db = DataBackend('combined', ['tushare','akshare'])
df = db._fetch_kline_tencent('600519', 'day', 60)
print(f'腾讯K线: {len(df)} bars, cols={list(df.columns)}')
print(df.tail(3))
"
```

- [ ] **Step 4: Commit**

```bash
git add src/stockagent_analysis/data_backend.py
git commit -m "feat: 腾讯K线直调(P3.5)降级源"
```

---

### Task 2: Tushare 作为 PE/PB 修正层（Snapshot增强）

**Files:**
- Modify: `src/stockagent_analysis/data_backend.py:220-226` (`collect_and_save_context` snapshot 部分)
- Modify: `src/stockagent_analysis/data_backend.py` (新增 `_correct_snapshot_pe_pb` 方法)

**说明:** Snapshot先从TDX/AKShare拿到close/pct_chg保覆盖率，再用Tushare daily_basic修正PE/PB。TDX的snapshot天然缺PE/PB（本地无此数据），AKShare历史行情也无PE。

- [ ] **Step 1: 新增 `_correct_snapshot_pe_pb` 方法**

在 `_fetch_snapshot_tdx` 方法之后（约 line 662），插入：

```python
def _correct_snapshot_pe_pb(self, snap: MarketSnapshot) -> MarketSnapshot:
    """用 Tushare daily_basic 修正/补充 PE_TTM 和换手率。
    Tushare的PE(TTM)数据最权威，用于修正TDX/AKShare缺失或异常的估值字段。
    """
    if not self._tushare_token:
        return snap
    try:
        import tushare as ts
        _tushare_throttle()
        ts.set_token(self._tushare_token)
        pro = ts.pro_api(timeout=self._tushare_timeout)
        ts_code = self._to_ts_code(snap.symbol)
        basic = pro.daily_basic(ts_code=ts_code, limit=1)
        if basic is not None and not basic.empty:
            row = basic.iloc[0]
            pe = self._safe_float(row.get("pe_ttm"))
            turnover = self._safe_float(row.get("turnover_rate"))
            # 仅在缺失或异常时修正
            if snap.pe_ttm is None or (pe is not None and abs(snap.pe_ttm) > 2000):
                snap = MarketSnapshot(
                    symbol=snap.symbol, name=snap.name,
                    close=snap.close, pct_chg=snap.pct_chg,
                    pe_ttm=pe if pe is not None else snap.pe_ttm,
                    turnover_rate=turnover if turnover is not None else snap.turnover_rate,
                    source=snap.source,
                )
            elif snap.turnover_rate is None and turnover is not None:
                snap = MarketSnapshot(
                    symbol=snap.symbol, name=snap.name,
                    close=snap.close, pct_chg=snap.pct_chg,
                    pe_ttm=snap.pe_ttm,
                    turnover_rate=turnover,
                    source=snap.source,
                )
    except Exception:
        pass  # 修正失败不影响主流程
    return snap
```

- [ ] **Step 2: 在 `collect_and_save_context` 中调用修正**

在 line 224（`snapshot = self._snapshot_from_klines(...)` 之后），插入修正调用：

```python
# Tushare PE/PB 修正层
snapshot = self._correct_snapshot_pe_pb(snapshot)
```

即原代码：
```python
snapshot = self._retry_snapshot_fetch(symbol, name, preferred_sources, progress_cb=progress_cb)
if snapshot.source == "mock":
    snapshot = self._snapshot_from_klines(symbol, name, kline_bundle.get("day", {}).get("df")) or snapshot
```
改为：
```python
snapshot = self._retry_snapshot_fetch(symbol, name, preferred_sources, progress_cb=progress_cb)
if snapshot.source == "mock":
    snapshot = self._snapshot_from_klines(symbol, name, kline_bundle.get("day", {}).get("df")) or snapshot
# Tushare PE/PB 修正层：先拿数据保覆盖率，再用权威源修正估值
snapshot = self._correct_snapshot_pe_pb(snapshot)
```

- [ ] **Step 3: 验证**

```bash
python -c "
from src.stockagent_analysis.data_backend import DataBackend, MarketSnapshot
db = DataBackend('combined', ['tushare','akshare'])
snap = MarketSnapshot('600519','贵州茅台',1850.0,1.5,None,None,'tdx')
corrected = db._correct_snapshot_pe_pb(snap)
print(f'Before: pe={snap.pe_ttm}, turnover={snap.turnover_rate}')
print(f'After:  pe={corrected.pe_ttm}, turnover={corrected.turnover_rate}')
"
```

- [ ] **Step 4: Commit**

```bash
git add src/stockagent_analysis/data_backend.py
git commit -m "feat: Tushare PE/PB修正层 — snapshot先拿数据再修正估值"
```

---

### Task 3: TDX感知的远程API信号量

**Files:**
- Modify: `src/stockagent_analysis/data_backend.py:94-102` (`__init__`)
- Modify: `src/stockagent_analysis/data_backend.py:384-438` (`_fetch_multi_timeframe_klines`)

**说明:** TDX本地读取不占远程API信号量，远程请求(AKShare/Tushare/腾讯)共享 `Semaphore(3)` 避免限流。当前是串行逐个尝试source，改为：先尝试TDX（快速本地），TDX失败后获取信号量再走远程。

- [ ] **Step 1: 在 `__init__` 中初始化信号量**

在 `data_backend.py:102`（`_ensure_akshare_no_proxy()` 之后），添加：

```python
import threading
self._remote_api_sem = threading.Semaphore(3)  # 远程数据API最多3路并发
```

- [ ] **Step 2: 修改 `_fetch_multi_timeframe_klines` 使用信号量**

在内层 source 循环中，对非TDX的source获取信号量：

将 `data_backend.py:406-422` 的 source 循环改为：

```python
for source in ordered_sources:
    if progress_cb:
        progress_cb("K线抓取", f"{tf} 第{used_attempts}/3次 source={source}")
    try:
        if source == "tdx":
            # TDX本地读取，不占远程信号量
            data = self._fetch_kline_tdx(symbol, tf, limit)
        else:
            # 远程API调用，受信号量控速
            with self._remote_api_sem:
                if source == "akshare":
                    data = self._fetch_kline_akshare(symbol, tf, limit)
                elif source == "tushare":
                    data = self._fetch_kline_tushare(symbol, tf, limit)
                elif source == "tencent":
                    data = self._fetch_kline_tencent(symbol, tf, limit)
                else:
                    raise ValueError(f"unsupported source: {source}")
        if data is not None and not data.empty and len(data) >= 1:
            ok = True
            used_source = source
            break
        last_error = "empty_or_insufficient_rows"
    except Exception as exc:
        last_error = str(exc)
```

- [ ] **Step 3: Commit**

```bash
git add src/stockagent_analysis/data_backend.py
git commit -m "feat: TDX感知信号量 — 本地读取不占远程API并发位"
```

---

### Task 4: 盘中虚拟K线（交易时段自动追加今日bar）

**Files:**
- Modify: `src/stockagent_analysis/data_backend.py` (新增 `_is_trading_session`, `_make_virtual_bar`, `_append_virtual_bar`)
- Modify: `src/stockagent_analysis/data_backend.py:175` (`collect_and_save_context` K线抓取后)

**说明:** 交易时段(9:15-15:05)运行分析时，从AKShare实时行情构造今日虚拟日K线追加到日线末尾，让盘中分析使用当天数据。TDX的 `_patch_intraday_as_daily` 已有类似逻辑但仅用于TDX源，此处为AKShare/Tushare源补充。

- [ ] **Step 1: 新增工具函数**

在 `_tushare_throttle` 函数之后（约 line 48），插入模块级函数：

```python
def _is_trading_session() -> bool:
    """判断当前是否在A股交易时段（工作日 9:15-15:05）。"""
    now = datetime.now()
    if now.weekday() >= 5:  # 周六日
        return False
    t = now.hour * 100 + now.minute
    return 915 <= t <= 1505
```

- [ ] **Step 2: 新增 `_append_virtual_bar_to_day` 方法**

在 `_correct_snapshot_pe_pb` 方法之后，插入：

```python
def _append_virtual_bar_to_day(self, day_df, symbol: str):
    """盘中模式：若日线最后一根不是今天，用AKShare实时行情构造虚拟bar追加。"""
    import pandas as pd

    if day_df is None or day_df.empty:
        return day_df
    if not _is_trading_session():
        return day_df

    today_str = datetime.now().strftime("%Y-%m-%d")
    last_ts = str(day_df.iloc[-1].get("ts", ""))[:10]
    if last_ts == today_str:
        return day_df  # 已包含今天

    try:
        import akshare as ak
        spot = ak.stock_zh_a_spot_em()
        row = spot[spot["代码"] == symbol]
        if row.empty:
            return day_df
        r = row.iloc[0]
        v_open = float(r.get("今开", 0) or 0)
        v_close = float(r.get("最新价", 0) or 0)
        v_high = float(r.get("最高", 0) or 0)
        v_low = float(r.get("最低", 0) or 0)
        v_vol = float(r.get("成交量", 0) or 0)
        v_amount = float(r.get("成交额", 0) or 0)
        if v_open <= 0 or v_close <= 0 or v_vol <= 0:
            return day_df  # 停牌/未开盘

        prev_close = float(day_df.iloc[-1]["close"])
        pct = round((v_close / prev_close - 1) * 100, 4) if prev_close > 0 else 0.0

        vbar = pd.DataFrame([{
            "ts": today_str,
            "open": v_open, "high": v_high, "low": v_low, "close": v_close,
            "volume": v_vol, "amount": v_amount, "pct_chg": pct,
        }])
        return pd.concat([day_df, vbar], ignore_index=True)
    except Exception:
        return day_df
```

- [ ] **Step 3: 在 `collect_and_save_context` 中调用**

在 K线bundle构建之后、保存CSV之前（约 line 183-184 之间），对日线追加虚拟bar：

```python
# 盘中模式：AKShare/Tushare源的日线追加今日虚拟bar
if kline_bundle.get("day", {}).get("ok") and kline_bundle["day"].get("source") != "tdx":
    day_df = kline_bundle["day"].get("df")
    if day_df is not None:
        patched = self._append_virtual_bar_to_day(day_df, symbol)
        if patched is not None and len(patched) > len(day_df):
            kline_bundle["day"]["df"] = patched
            kline_bundle["day"]["rows"] = len(patched)
```

注：TDX源已有 `_patch_intraday_as_daily` 处理，此处仅补充非TDX源。

- [ ] **Step 4: Commit**

```bash
git add src/stockagent_analysis/data_backend.py
git commit -m "feat: 盘中虚拟K线 — 交易时段自动追加今日实时bar"
```

---

### Task 5: Snapshot增加腾讯/新浪实时行情降级

**Files:**
- Modify: `src/stockagent_analysis/data_backend.py:440-447` (`_fetch_from_source`)
- Modify: `src/stockagent_analysis/data_backend.py` (新增 `_fetch_snapshot_tencent`)

**说明:** 当前snapshot只有 TDX→AKShare→Tushare 三个源。增加腾讯实时行情作为额外降级。

- [ ] **Step 1: 新增 `_fetch_snapshot_tencent` 方法**

在 `_fetch_snapshot_tdx` 方法之后插入：

```python
def _fetch_snapshot_tencent(self, symbol: str, name: str) -> MarketSnapshot:
    """腾讯实时行情快照（qt.gtimg.cn），作为AKShare东财接口挂掉时的备选。"""
    import requests as _req

    market_code = f"sh{symbol}" if symbol.startswith("6") else f"sz{symbol}"
    resp = _req.get(f"https://qt.gtimg.cn/q={market_code}", timeout=10)
    resp.encoding = "gbk"
    text = resp.text.strip()
    if "=" not in text:
        raise RuntimeError("tencent_snapshot_parse_error")
    val_part = text.split("=", 1)[1].strip().strip('"')
    if not val_part:
        raise RuntimeError("tencent_snapshot_empty")
    fields = val_part.split("~")
    if len(fields) < 47:
        raise RuntimeError("tencent_snapshot_fields_insufficient")

    current = self._safe_float(fields[3]) or 0.0
    yesterday = self._safe_float(fields[4]) or 0.0
    pct = round((current - yesterday) / yesterday * 100, 4) if yesterday > 0 else 0.0
    pe = self._safe_float(fields[39]) if len(fields) > 39 else None
    turnover = self._safe_float(fields[38]) if len(fields) > 38 else None

    if current <= 0:
        raise RuntimeError("tencent_snapshot_no_price")

    return MarketSnapshot(
        symbol=symbol,
        name=fields[1] if len(fields) > 1 else name,
        close=current,
        pct_chg=pct,
        pe_ttm=pe,
        turnover_rate=turnover,
        source="tencent",
    )
```

- [ ] **Step 2: 修改 `_fetch_from_source` 增加 tencent 分支**

`data_backend.py:440-447` 改为：

```python
def _fetch_from_source(self, source: str, symbol: str, name: str) -> MarketSnapshot:
    if source == "tdx":
        return self._fetch_snapshot_tdx(symbol, name)
    if source == "akshare":
        return self._fetch_akshare(symbol, name)
    if source == "tushare":
        return self._fetch_tushare(symbol, name)
    if source == "tencent":
        return self._fetch_snapshot_tencent(symbol, name)
    raise ValueError(f"unsupported source: {source}")
```

- [ ] **Step 3: 修改 `fetch_snapshot` 确保 tencent 在降级链中**

`data_backend.py:124-139`，在构建 sources 列表时追加 tencent 作为最终兜底：

```python
def fetch_snapshot(self, symbol: str, name: str, preferred_sources: list[str] | None = None) -> MarketSnapshot:
    symbol = self._clean_symbol(symbol)
    sources = list(preferred_sources or self.default_sources)
    if "tdx" not in sources:
        sources = ["tdx"] + sources
    # 确保腾讯行情作为最终兜底
    if "tencent" not in sources:
        sources.append("tencent")
    if self.mode == "single":
        return self._fetch_from_source(sources[0], symbol, name)
    for src in sources:
        try:
            return self._fetch_from_source(src, symbol, name)
        except Exception:
            continue
    return MarketSnapshot(symbol=symbol, name=name, close=0.0, pct_chg=0.0, pe_ttm=None, turnover_rate=None, source="mock")
```

- [ ] **Step 4: Commit**

```bash
git add src/stockagent_analysis/data_backend.py
git commit -m "feat: Snapshot增加腾讯实时行情(P4)降级源"
```

---

### Task 6: 集成验证

- [ ] **Step 1: 语法检查**

```bash
python -c "import ast; ast.parse(open('src/stockagent_analysis/data_backend.py', encoding='utf-8').read()); print('OK')"
```

- [ ] **Step 2: Import检查**

```bash
python -c "from src.stockagent_analysis.data_backend import DataBackend; print('import OK')"
```

- [ ] **Step 3: 端到端运行验证**

```bash
python run.py analyze --symbol 002571 --name 索菲亚 --providers deepseek,grok
```

检查：
- 无 Traceback
- K线 meta.json 中 source 字段正确（优先显示 tdx）
- Snapshot pe_ttm 非 None（Tushare修正层生效）
- 输出PDF和final_decision正常

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: 数据获取梯次化增强 — 5级K线降级+PE修正+TDX信号量+盘中虚拟K线"
```

---

## 改进后数据源优先级总结

### K线
```
P1: TDX本地（不占信号量，最快）
P2: AKShare东财（占信号量）
P3: Tushare（占信号量，需Token）
P3.5: 腾讯K线直调（占信号量，仅日线，独立数据源）
```

### Snapshot
```
P1: TDX本地（close/pct_chg, 无PE）
P2: AKShare东财（close/pct_chg/PE/换手率）
P3: Tushare（close/pct_chg/PE/换手率）
P4: 腾讯实时行情（close/pct_chg/PE/换手率）
→ 最终: Tushare修正层（修正PE/PB异常值）
```
