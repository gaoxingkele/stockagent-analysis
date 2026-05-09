"""V11 LLM 视觉过滤模块 (从 v12_llm_filter_0508.py 抽出).

主要类:
  V11VisionFilter
    - filter_one(ts_code, date)         单股 LLM 视觉
    - filter_batch(symbols, date, cb)   批量带进度回调

依赖:
  - output/tushare_cache/daily/{YYYYMMDD}.parquet (565 日全市场 OHLC)
  - chart_generator (生成 PNG)
  - cloubic API (claude-sonnet-4-6)

进度回调签名:
  cb(phase: str, percent: int, message: str, data: dict | None) -> None
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from openai import OpenAI

from .chart_generator import generate_kline_chart

ProgressCb = Optional[Callable[[str, int, str, Optional[dict]], None]]

CLOUBIC_BASE = "https://api.cloubic.com/v1"
LLM_MODEL = "claude-sonnet-4-6"

SYS_PROMPT_V11 = """你是 A 股专业 K 线视觉技术分析师, 严格基于图像本身做客观分析 (不依赖外部信息).

你将看到同一只股票的 3 张 K 线综合图 (按顺序: 月线 / 周线 / 日线), 每张图含蜡烛+MA5/10/20/60/120/250+布林带+趋势线+成交量+MACD+RSI+KDJ.

## 分析框架 (必须覆盖 5 大域)

### 1. 趋势分析 (Dow Theory)
- 主要趋势 (月线): 上升/下降/震荡, 强度
- 次要趋势 (周线): 是否与主要趋势同向?
- 短期趋势 (日线): 当前位置
- HH/HL pattern (Higher High/Higher Low) 是否完整?

### 2. 支撑/阻力 (3+ touches 才算 strong)
- 月线/周线 关键长期支撑阻力位
- 日线 短期支撑阻力位

### 3. 移动均线
- 价格 vs MA20/60/250 关系 (above/below/between)
- 金叉/死叉 (Golden Cross / Death Cross)
- MA 排列 (bullish 多头排列 / bearish 空头排列)

### 4. 成交量
- 量价配合 (rising price + rising volume = healthy)
- 量价背离 (rising price + falling volume = weak)

### 5. 形态 + 缠论 + Elliott Wave
- 反转/延续形态: hammer/engulfing/双底/M 头/W 底/bull flag/triangle
- 缠论视角: 顶分型/底分型? 中枢震荡/突破? 一类/二类/三类买点?
- Elliott Wave: 主升浪 (1/3/5) 还是修正浪 (a/b/c)? Fibonacci 位

## 输出 JSON (不要 markdown 包裹):
{
  "trend_main": "uptrend/downtrend/sideways",
  "trend_strength": "strong/moderate/weak",
  "ma_alignment": "bullish/bearish/neutral",
  "volume_confirms_price": true/false,
  "key_pattern": "30字内核心形态描述, 含缠论术语",
  "elliott_phase": "30字内 Elliott Wave 当前位置",
  "scenarios": [
    {"name": "Bull", "prob": 0.0-1.0, "target_pct_20d": +/- 数值, "invalidation": "30字内"},
    {"name": "Base", "prob": 0.0-1.0, "target_pct_20d": +/- 数值, "invalidation": "30字内"},
    {"name": "Bear", "prob": 0.0-1.0, "target_pct_20d": +/- 数值, "invalidation": "30字内"}
  ]
}

scenarios 概率求和必须等于 1.0. 注意: 你不知道 V7c 量化系统对这只股的看法, 完全独立判断."""


def _parse_json_robust(text: str) -> Optional[dict]:
    if not text: return None
    text = text.strip()
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m: text = m.group(1)
    else:
        m = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if m: text = m.group(0)
    try: return json.loads(text)
    except Exception: return None


def _extract_scenarios(j: dict) -> tuple:
    bull_p = base_p = bear_p = float("nan")
    bull_t = base_t = bear_t = float("nan")
    for s in j.get("scenarios", []):
        n = str(s.get("name", "")).lower()
        try: p = float(s.get("prob", 0))
        except Exception: p = float("nan")
        try: t = float(s.get("target_pct_20d", 0))
        except Exception: t = float("nan")
        if "bull" in n: bull_p, bull_t = p, t
        elif "base" in n: base_p, base_t = p, t
        elif "bear" in n: bear_p, bear_t = p, t
    return bull_p, base_p, bear_p, bull_t, base_t, bear_t


class V11VisionFilter:
    """V11 LLM 视觉过滤器 (单例; daily cache + LLM client 复用)."""

    _instance: Optional["V11VisionFilter"] = None

    def __init__(self, project_root: Path, cloubic_api_key: str):
        self.root = Path(project_root)
        self.daily_cache = self.root / "output" / "tushare_cache" / "daily"
        self.charts_dir = self.root / "output" / "v12" / "charts"
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self._daily: Optional[dict] = None
        self.client = OpenAI(api_key=cloubic_api_key, base_url=CLOUBIC_BASE)

    @classmethod
    def get(cls, project_root: Path, cloubic_api_key: str) -> "V11VisionFilter":
        if cls._instance is None:
            cls._instance = cls(project_root, cloubic_api_key)
        return cls._instance

    # ──────── 数据 ────────
    def _load_daily_cache(self, end_date: str, cb: ProgressCb = None):
        if self._daily is not None: return self._daily
        if cb: cb("load_daily", 2, f"加载 Tushare daily cache (565 文件) ...", None)
        files = sorted(self.daily_cache.glob("*.parquet"))
        end_int = int(end_date)
        parts = [pd.read_parquet(f) for f in files if int(f.stem) <= end_int]
        big = pd.concat(parts, ignore_index=True)
        big["trade_date"] = big["trade_date"].astype(str)
        big = big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        self._daily = {ts: g.reset_index(drop=True) for ts, g in big.groupby("ts_code")}
        if cb: cb("load_daily", 5, f"daily cache 加载完成: {len(self._daily)} 股", None)
        return self._daily

    def _get_ohlc(self, ts_code: str, end_date: str) -> Optional[list[dict]]:
        cache = self._daily or {}
        df = cache.get(ts_code)
        if df is None: return None
        df = df[df["trade_date"] <= end_date]
        if df.empty: return None
        rows = []
        for _, r in df.iterrows():
            td = r["trade_date"]
            ts = f"{td[:4]}-{td[4:6]}-{td[6:8]}"
            rows.append({
                "ts": ts,
                "open": float(r["open"]), "high": float(r["high"]),
                "low": float(r["low"]), "close": float(r["close"]),
                "volume": float(r.get("vol", r.get("volume", 0))),
            })
        return rows

    def _daily_to_periodic(self, rows: list[dict], freq: str) -> list[dict]:
        if not rows: return []
        df = pd.DataFrame(rows)
        df["dt"] = pd.to_datetime(df["ts"])
        df = df.set_index("dt")
        if freq == "W":
            agg = df.resample("W-FRI").agg({"open": "first", "high": "max", "low": "min",
                                             "close": "last", "volume": "sum"}).dropna()
        elif freq == "M":
            agg = df.resample("ME").agg({"open": "first", "high": "max", "low": "min",
                                          "close": "last", "volume": "sum"}).dropna()
        else:
            return rows
        agg["ts"] = agg.index.strftime("%Y-%m-%d")
        return agg[["ts", "open", "high", "low", "close", "volume"]].to_dict("records")

    def _gen_3tf_charts(self, ts_code: str, date: str) -> tuple:
        rows = self._get_ohlc(ts_code, date)
        if not rows or len(rows) < 60:
            return None, None, None
        paths = {}
        for tf, freq, take_n in [("month", "M", 120), ("week", "W", 60), ("day", None, 60)]:
            target = self.charts_dir / f"{ts_code}_{date}_{tf}.png"
            if target.exists():
                paths[tf] = target; continue
            data = self._daily_to_periodic(rows, freq) if freq else rows[-take_n:]
            if not data or len(data) < 10:
                paths[tf] = None; continue
            data = data[-take_n:]
            df = pd.DataFrame(data)
            png = generate_kline_chart(df, tf, ts_code, "?", save_path=target)
            paths[tf] = target if png else None
        return paths.get("month"), paths.get("week"), paths.get("day")

    @staticmethod
    def _b64(path: Path) -> str:
        return base64.b64encode(path.read_bytes()).decode("utf-8")

    # ──────── LLM 调用 ────────
    def _call_vision(self, m_b64: str, w_b64: str, d_b64: str, max_retry: int = 2):
        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{m_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{w_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{d_b64}"}},
            {"type": "text", "text": "上面三张图按顺序是: 月线 (10 年)、周线 (60 周)、日线 (60 日). 请按 5 域框架做分析, 输出 JSON."},
        ]
        for r in range(max_retry):
            try:
                resp = self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": SYS_PROMPT_V11},
                        {"role": "user", "content": user_content},
                    ],
                    max_tokens=600, temperature=0.0,
                )
                return resp.choices[0].message.content, resp.usage
            except Exception:
                if r < max_retry - 1:
                    time.sleep(3); continue
                return None, None
        return None, None

    # ──────── 公开接口 ────────
    def filter_one(self, ts_code: str, date: str) -> dict:
        """单股 V11 视觉过滤. 返回 {status, bull_prob, ...}."""
        self._load_daily_cache(date)
        m_p, w_p, d_p = self._gen_3tf_charts(ts_code, date)
        if not (m_p and w_p and d_p):
            return {"status": "no_image", "ts_code": ts_code,
                    "bull_prob": np.nan, "base_prob": np.nan, "bear_prob": np.nan}
        text, usage = self._call_vision(self._b64(m_p), self._b64(w_p), self._b64(d_p))
        if text is None:
            return {"status": "llm_error", "ts_code": ts_code,
                    "bull_prob": np.nan, "base_prob": np.nan, "bear_prob": np.nan,
                    "tokens_in": 0, "tokens_out": 0}
        j = _parse_json_robust(text)
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0
        if j is None:
            return {"status": "parse_error", "ts_code": ts_code,
                    "bull_prob": np.nan, "base_prob": np.nan, "bear_prob": np.nan,
                    "raw_text": (text or "")[:200],
                    "tokens_in": tokens_in, "tokens_out": tokens_out}
        bp, bsp, brp, bt, bst, brt = _extract_scenarios(j)
        return {
            "status": "ok", "ts_code": ts_code,
            "bull_prob": bp, "base_prob": bsp, "bear_prob": brp,
            "bull_target": bt, "base_target": bst, "bear_target": brt,
            "trend_main": str(j.get("trend_main", ""))[:30],
            "trend_strength": str(j.get("trend_strength", ""))[:20],
            "ma_alignment": str(j.get("ma_alignment", ""))[:20],
            "key_pattern": str(j.get("key_pattern", ""))[:120],
            "elliott_phase": str(j.get("elliott_phase", ""))[:80],
            "tokens_in": tokens_in, "tokens_out": tokens_out,
        }

    def filter_batch(self, symbols: list[str], date: str,
                     cb: ProgressCb = None) -> list[dict]:
        """批量过滤. cb 在每股完成后调用."""
        self._load_daily_cache(date, cb=cb)
        results = []
        n = len(symbols)
        cost_sum = 0.0
        for i, ts in enumerate(symbols, 1):
            t0 = time.time()
            rec = self.filter_one(ts, date)
            results.append(rec)
            ti = rec.get("tokens_in", 0); to = rec.get("tokens_out", 0)
            cost_sum += (ti * 3.0 + to * 15.0) / 1e6
            if cb:
                pct = 5 + int(95 * i / n)
                bp = rec.get("bull_prob", float("nan"))
                bp_s = f"{bp:.2f}" if not (isinstance(bp, float) and np.isnan(bp)) else "N/A"
                cb("llm_filter", pct,
                   f"[{i}/{n}] {ts} status={rec['status']} bull={bp_s} ${cost_sum:.3f}",
                   {"i": i, "n": n, "ts_code": ts, "status": rec["status"],
                    "bull_prob": None if (isinstance(bp, float) and np.isnan(bp)) else bp,
                    "duration_sec": time.time() - t0,
                    "cost_usd_sum": round(cost_sum, 4)})
        return results
