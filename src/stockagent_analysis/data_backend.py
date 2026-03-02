# -*- coding: utf-8 -*-
import os
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv


load_dotenv()

# 数据源域名（Tushare + AKShare），统一加入 NO_PROXY，确保数据采集不走代理
_DATA_NO_PROXY = (
    "api.tushare.pro,"
    "push2his.eastmoney.com,quote.eastmoney.com,"
    "finance.sina.com.cn,gu.qq.com"
)
# 兼容旧引用
_AKSHARE_NO_PROXY = _DATA_NO_PROXY

# Tushare 调用节流：5000积分常规数据 500次/分，分钟数据需单独开通权限。间隔秒数可配 TUSHARE_CALL_INTERVAL_SEC
_TUSHARE_LAST_CALL: float = 0.0


def _tushare_throttle() -> None:
    """Tushare 调用节流，避免触发限频（参考 https://tushare.pro/document/1?doc_id=290）。"""
    global _TUSHARE_LAST_CALL
    interval = float(os.getenv("TUSHARE_CALL_INTERVAL_SEC", "5.0"))
    if interval <= 0:
        return
    elapsed = time.time() - _TUSHARE_LAST_CALL
    if elapsed < interval:
        time.sleep(interval - elapsed)
    _TUSHARE_LAST_CALL = time.time()


def _ensure_akshare_no_proxy() -> None:
    """确保数据源域名（Tushare/AKShare）在 NO_PROXY 中，数据采集不走代理。"""
    current = os.environ.get("NO_PROXY", "").strip()
    if not current:
        os.environ["NO_PROXY"] = _DATA_NO_PROXY
    else:
        existing = {d.strip() for d in current.split(",") if d.strip()}
        to_add = [d.strip() for d in _DATA_NO_PROXY.split(",") if d.strip() and d.strip() not in existing]
        if to_add:
            os.environ["NO_PROXY"] = f"{current},{','.join(to_add)}".strip(",")
    os.environ["no_proxy"] = os.environ["NO_PROXY"]


@dataclass
class MarketSnapshot:
    symbol: str
    name: str
    close: float
    pct_chg: float
    pe_ttm: float | None
    turnover_rate: float | None
    source: str


class DataBackend:
    def __init__(self, mode: str, default_sources: list[str]):
        self.mode = mode
        self.default_sources = default_sources
        self._tushare_token = os.getenv("TUSHARE_TOKEN", "").strip()
        self._tushare_timeout = int(os.getenv("TUSHARE_TIMEOUT_SEC", "15"))
        _ensure_akshare_no_proxy()

    def fetch_snapshot(self, symbol: str, name: str, preferred_sources: list[str] | None = None) -> MarketSnapshot:
        symbol = self._clean_symbol(symbol)
        sources = preferred_sources or self.default_sources
        if self.mode == "single":
            return self._fetch_from_source(sources[0], symbol, name)
        # combined: 按顺序尝试，成功即返回
        for src in sources:
            try:
                return self._fetch_from_source(src, symbol, name)
            except Exception:
                continue
        # 最终兜底
        return MarketSnapshot(symbol=symbol, name=name, close=0.0, pct_chg=0.0, pe_ttm=None, turnover_rate=None, source="mock")

    def collect_and_save_context(
        self,
        symbol: str,
        name: str,
        run_dir: str,
        preferred_sources: list[str] | None = None,
        progress_cb=None,
    ) -> dict[str, Any]:
        """抓取并落地可校验的数据包：多级别K线、消息、基本面、因子。"""
        import json
        from pathlib import Path

        # 标准化：去除 .SZ/.SH 等后缀，确保传给 AKShare 的是纯6位数字代码
        symbol = self._clean_symbol(symbol)

        _ensure_akshare_no_proxy()

        data_dir = Path(run_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        kline_dir = data_dir / "kline"
        kline_dir.mkdir(parents=True, exist_ok=True)

        sources = preferred_sources or self.default_sources
        # 技术/量价/结构分析采用日线、周线、月线；1h供视觉智能体使用
        timeframes = {
            "month": 60,
            "week": 100,
            "day": 100,
        }
        # 1h单独抓取（分钟数据权限要求不同，失败不影响主流程）
        timeframes_1h = {"1h": 160}

        if progress_cb:
            progress_cb("K线数据抓取", "开始")
        kline_bundle = self._fetch_multi_timeframe_klines(symbol, sources, timeframes, progress_cb=progress_cb)
        # 1h K线：失败不影响主流程
        try:
            bundle_1h = self._fetch_multi_timeframe_klines(symbol, sources, timeframes_1h, progress_cb=progress_cb)
            kline_bundle.update(bundle_1h)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("1h kline fetch failed (non-fatal): %s", e)
            kline_bundle["1h"] = {"ok": False, "partial": False, "df": None, "rows": 0, "source": "unknown", "attempts": 1, "error": str(e)}
        for tf, item in kline_bundle.items():
            (kline_dir / f"{tf}.meta.json").write_text(
                json.dumps(
                    {
                        "timeframe": tf,
                        "source": item.get("source"),
                        "attempts": item.get("attempts"),
                        "ok": item.get("ok"),
                        "partial": item.get("partial", False),
                        "error": item.get("error"),
                        "rows": item.get("rows"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            df = item.get("df")
            if df is not None and not df.empty:
                df.to_csv(kline_dir / f"{tf}.csv", index=False, encoding="utf-8-sig")
            else:
                (kline_dir / f"{tf}.csv").write_text(
                    "ts,open,high,low,close,volume,amount,pct_chg\n",
                    encoding="utf-8-sig",
                )
        day_df = kline_bundle["day"].get("df")
        if day_df is not None and not day_df.empty:
            day_df.to_csv(data_dir / "historical_daily.csv", index=False, encoding="utf-8-sig")
        else:
            (data_dir / "historical_daily.csv").write_text(
                "ts,open,high,low,close,volume,amount,pct_chg\n",
                encoding="utf-8-sig",
            )
        if progress_cb:
            progress_cb("K线数据抓取", "完成")

        if progress_cb:
            progress_cb("行情快照抓取", "开始")
        snapshot = self._retry_snapshot_fetch(symbol, name, preferred_sources, progress_cb=progress_cb)
        if snapshot.source == "mock":
            snapshot = self._snapshot_from_klines(symbol, name, kline_bundle.get("day", {}).get("df")) or snapshot
        if progress_cb:
            progress_cb("行情快照抓取", f"完成 source={snapshot.source}")

        if progress_cb:
            progress_cb("基本面数据抓取", "开始")
        fundamentals = self._retry_fetch(
            fetch_name="基本面数据抓取",
            progress_cb=progress_cb,
            checker=lambda x: isinstance(x, dict) and x.get("source") != "unknown",
            func=lambda: self._fetch_fundamentals(symbol),
            fallback={"source": "unknown"},
        )
        (data_dir / "fundamentals.json").write_text(
            json.dumps(fundamentals, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if progress_cb:
            progress_cb("基本面数据抓取", f"完成 source={fundamentals.get('source', 'unknown')}")

        if progress_cb:
            progress_cb("新闻数据抓取", "开始")
        news = self._retry_fetch(
            fetch_name="新闻数据抓取",
            progress_cb=progress_cb,
            checker=lambda x: isinstance(x, list) and len(x) > 0,
            func=lambda: self._fetch_news(symbol),
            fallback=[],
        )
        (data_dir / "news.json").write_text(
            json.dumps(news, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if progress_cb:
            progress_cb("新闻数据抓取", f"完成 count={len(news)}")

        if progress_cb:
            progress_cb("因子与指标计算", "开始")
        features = self._compute_features(snapshot, day_df, fundamentals, news, kline_bundle)
        (data_dir / "factors.json").write_text(
            json.dumps(features, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if progress_cb:
            progress_cb("因子与指标计算", "完成")

        # 生成多周期K线图（供视觉智能体使用）
        chart_files: dict[str, str] = {}
        if progress_cb:
            progress_cb("K线图表生成", "开始")
        try:
            from .chart_generator import generate_all_charts
            charts_dir = data_dir / "charts"
            chart_files = generate_all_charts(kline_bundle, symbol, name, charts_dir)
            if progress_cb:
                progress_cb("K线图表生成", f"完成 生成{len(chart_files)}张图 {list(chart_files.keys())}")
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("chart generation failed (non-fatal): %s", e)
            if progress_cb:
                progress_cb("K线图表生成", f"失败(非致命): {e}")

        required_tfs = ["day", "week", "month"]
        kline_ok = all(kline_bundle.get(k, {}).get("ok") for k in required_tfs)
        fundamentals_ok = fundamentals.get("source") != "unknown"
        news_ok = len(news) > 0
        snapshot_ok = snapshot.source != "mock"
        is_complete = bool(kline_ok and fundamentals_ok and news_ok and snapshot_ok)
        data_integrity = {
            "is_complete": is_complete,
            "snapshot_ok": snapshot_ok,
            "kline_ok": kline_ok,
            "fundamentals_ok": fundamentals_ok,
            "news_ok": news_ok,
            "failed_items": [k for k in timeframes if not kline_bundle[k]["ok"]],
        }

        context = {
            "meta": {
                "symbol": symbol,
                "name": name,
                "generated_at": datetime.now().isoformat(),
            },
            "snapshot": {
                "symbol": snapshot.symbol,
                "name": snapshot.name,
                "close": snapshot.close,
                "pct_chg": snapshot.pct_chg,
                "pe_ttm": snapshot.pe_ttm,
                "turnover_rate": snapshot.turnover_rate,
                "source": snapshot.source,
            },
            "fundamentals": fundamentals,
            "news": news[:30],
            "features": features,
            "data_integrity": data_integrity,
            "chart_files": chart_files,
            "data_files": {
                "historical_daily_csv": str((data_dir / "historical_daily.csv")),
                "fundamentals_json": str((data_dir / "fundamentals.json")),
                "news_json": str((data_dir / "news.json")),
                "factors_json": str((data_dir / "factors.json")),
                "kline_month_csv": str((kline_dir / "month.csv")),
                "kline_week_csv": str((kline_dir / "week.csv")),
                "kline_day_csv": str((kline_dir / "day.csv")),
                **{f"kline_{tf}_png": path for tf, path in chart_files.items()},
            },
        }
        (data_dir / "analysis_context.json").write_text(
            json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return context

    def _retry_snapshot_fetch(
        self,
        symbol: str,
        name: str,
        preferred_sources: list[str] | None = None,
        progress_cb=None,
    ) -> MarketSnapshot:
        return self._retry_fetch(
            fetch_name="行情快照抓取",
            progress_cb=progress_cb,
            checker=lambda x: isinstance(x, MarketSnapshot) and x.source != "mock",
            func=lambda: self.fetch_snapshot(symbol, name, preferred_sources),
            fallback=MarketSnapshot(symbol=symbol, name=name, close=0.0, pct_chg=0.0, pe_ttm=None, turnover_rate=None, source="mock"),
        )

    @staticmethod
    def _retry_fetch(fetch_name: str, progress_cb, checker, func, fallback):
        last = fallback
        for i in range(1, 4):
            if progress_cb:
                progress_cb(fetch_name, f"第{i}/3次")
            try:
                last = func()
                if checker(last):
                    return last
            except Exception:
                pass
        return last

    def _fetch_multi_timeframe_klines(
        self,
        symbol: str,
        sources: list[str],
        timeframes: dict[str, int],
        progress_cb=None,
    ) -> dict[str, dict[str, Any]]:
        """K线抓取：Tushare 优先，失败则改用 AKShare。每周期最多重试 3 次，每次尝试 Tushare→AKShare。"""
        # 确保顺序：Tushare 优先，AKShare 备选
        ordered_sources = ["tushare", "akshare"]
        for s in sources:
            if s not in ordered_sources:
                ordered_sources.append(s)
        bundle: dict[str, dict[str, Any]] = {}
        for tf, limit in timeframes.items():
            ok = False
            last_error = ""
            data = None
            used_source = "unknown"
            used_attempts = 0
            for attempt in range(3):
                used_attempts = attempt + 1
                for source in ordered_sources:
                    if progress_cb:
                        progress_cb("K线抓取", f"{tf} 第{used_attempts}/3次 source={source}")
                    try:
                        if source == "akshare":
                            data = self._fetch_kline_akshare(symbol, tf, limit)
                        elif source == "tushare":
                            data = self._fetch_kline_tushare(symbol, tf, limit)
                        else:
                            raise ValueError(f"unsupported source: {source}")
                        min_rows = min(limit, 30)
                        if data is not None and not data.empty and len(data) >= min_rows:
                            ok = True
                            used_source = source
                            break
                        last_error = "empty_or_insufficient_rows"
                    except Exception as exc:
                        last_error = str(exc)
                if ok:
                    break
            partial = ok and data is not None and not data.empty and len(data) < limit
            bundle[tf] = {
                "ok": ok,
                "partial": partial,
                "source": used_source if ok else "unknown",
                "attempts": used_attempts,
                "error": None if ok else last_error,
                "rows": int(len(data)) if data is not None else 0,
                "df": data.tail(limit).copy() if data is not None and not data.empty else None,
            }
        return bundle

    def _fetch_from_source(self, source: str, symbol: str, name: str) -> MarketSnapshot:
        if source == "akshare":
            return self._fetch_akshare(symbol, name)
        if source == "tushare":
            return self._fetch_tushare(symbol, name)
        raise ValueError(f"unsupported source: {source}")

    def _fetch_hist_akshare(self, symbol: str):
        import akshare as ak

        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
        for _ in range(2):
            try:
                df = ak.stock_zh_a_hist(
                    symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust=""
                )
                if df is not None and not df.empty:
                    return df
            except Exception:
                continue
        return None

    def _fetch_kline_akshare(self, symbol: str, timeframe: str, limit: int):
        import akshare as ak
        import pandas as pd

        if timeframe in {"day", "week", "month"}:
            period = "daily" if timeframe == "day" else ("weekly" if timeframe == "week" else "monthly")
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
            raw = None
            try:
                raw = ak.stock_zh_a_hist(
                    symbol=symbol, period=period, start_date=start_date, end_date=end_date, adjust=""
                )
            except Exception:
                pass
            # 东财失败时尝试腾讯数据源（gu.qq.com，可规避代理对东财的拦截）
            if raw is None or raw.empty:
                try:
                    ts_symbol = f"sz{symbol}" if symbol.startswith(("0", "3")) else f"sh{symbol}"
                    raw = ak.stock_zh_a_hist_tx(symbol=ts_symbol, adjust="")
                    if raw is not None and not raw.empty and timeframe in {"week", "month"}:
                        raw = raw.copy()
                        raw["date"] = pd.to_datetime(raw["date"])
                        resample_rule = "W-FRI" if timeframe == "week" else "ME"
                        raw = (
                            raw.set_index("date")
                            .resample(resample_rule)
                            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "amount": "sum"})
                            .dropna()
                            .reset_index()
                        )
                        raw["date"] = raw["date"].dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
            if raw is None or raw.empty:
                raise RuntimeError(f"akshare_{timeframe}_empty")
            return self._normalize_kline_df(raw).tail(limit)
        period = "60" if timeframe in {"1h", "2h"} else "30"
        raw = None
        # 东财分钟线：重试+延迟，避免限频导致 RemoteDisconnected
        for attempt in range(3):
            if attempt > 0:
                time.sleep(2 + attempt)
            try:
                end_dt = datetime.now().strftime("%Y-%m-%d 15:00:00")
                start_dt = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d 09:30:00")
                raw = ak.stock_zh_a_hist_min_em(
                    symbol=symbol, period=period, start_date=start_dt, end_date=end_dt, adjust=""
                )
                if raw is not None and not raw.empty:
                    break
            except Exception:
                pass
        # 东财失败时尝试新浪备选（finance.sina.com.cn，不同数据源）
        if raw is None or raw.empty:
            try:
                sina_symbol = f"sz{symbol}" if symbol.startswith(("0", "3")) else f"sh{symbol}"
                raw = ak.stock_zh_a_minute(symbol=sina_symbol, period=period, adjust="")
                if raw is not None and not raw.empty:
                    raw["day"] = raw["day"].astype(str)
            except Exception:
                pass
        if raw is None or raw.empty:
            raise RuntimeError(f"akshare_{timeframe}_empty")
        df = self._normalize_kline_df(raw).sort_values("ts")
        if timeframe == "2h":
            min_1h_rows = min(limit * 2, 60)  # 降级：至少 30 根 2h（60 根 1h）
            if len(df) < min_1h_rows:
                raise RuntimeError("akshare_2h_rows_not_enough")
            df = df.tail(min(len(df), limit * 2)).reset_index(drop=True)
            rows = []
            for idx in range(0, len(df), 2):
                g = df.iloc[idx : idx + 2]
                if g.empty:
                    continue
                rows.append(
                    {
                        "ts": g.iloc[-1]["ts"],
                        "open": float(g.iloc[0]["open"]),
                        "high": float(g["high"].max()),
                        "low": float(g["low"].min()),
                        "close": float(g.iloc[-1]["close"]),
                        "volume": float(g["volume"].sum()),
                        "amount": float(g["amount"].sum()),
                        "pct_chg": float((g.iloc[-1]["close"] / g.iloc[0]["open"] - 1.0) * 100) if g.iloc[0]["open"] else 0.0,
                    }
                )
            df = pd.DataFrame(rows)
        min_rows = min(limit, 30)
        if len(df) < min_rows:
            raise RuntimeError(f"akshare_{timeframe}_rows_not_enough")
        return df.tail(limit)

    def _fetch_kline_tushare(self, symbol: str, timeframe: str, limit: int):
        import tushare as ts

        if not self._tushare_token:
            raise RuntimeError("missing TUSHARE_TOKEN")
        ts.set_token(self._tushare_token)
        pro = ts.pro_api(timeout=self._tushare_timeout)
        ts_code = self._to_ts_code(symbol)
        freq_map = {
            "day": "D",
            "week": "W",
            "month": "M",
            "1h": "60min",
            "2h": "120min",
            "30m": "30min",
        }
        freq = freq_map[timeframe]
        last_err: Exception | None = None
        for retry in range(3):
            _tushare_throttle()
            try:
                df = ts.pro_bar(api=pro, ts_code=ts_code, adj=None, freq=freq, limit=limit + 30)
                if df is not None and not df.empty:
                    out = self._normalize_kline_df(df)
                    if len(out) >= limit:
                        return out.tail(limit)
                    last_err = RuntimeError(f"tushare_{timeframe}_rows_not_enough")
                else:
                    last_err = RuntimeError(f"tushare_{timeframe}_empty")
            except Exception as e:
                err_str = str(e)
                # 连接超时/网络错误：立即放弃 Tushare，让外层尝试 AKShare
                exc_name = type(e).__name__
                if "Timeout" in exc_name or "Connection" in exc_name:
                    raise
                cause = getattr(e, "__cause__", None)
                if cause and ("Timeout" in type(cause).__name__ or "Connection" in type(cause).__name__):
                    raise
                if "timed out" in err_str or "Connection to api.waditu.com" in err_str:
                    raise
                # 权限/限频：每天2次、每分钟2次等，立即改用 AKShare，不重试
                if "每天最多" in err_str or "每分钟最多" in err_str or "限频" in err_str or "4003" in err_str or "429" in err_str:
                    raise
                raise
            break
        if last_err:
            raise last_err
        raise RuntimeError(f"tushare_{timeframe}_empty")

    @staticmethod
    def _normalize_kline_df(df):
        import pandas as pd

        if df is None or df.empty:
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume", "amount", "pct_chg"])
        col_map = {
            "日期": "ts",
            "时间": "ts",
            "date": "ts",
            "day": "ts",
            "trade_date": "ts",
            "trade_time": "ts",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "涨跌幅": "pct_chg",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "vol": "volume",
            "amount": "amount",
            "pct_chg": "pct_chg",
        }
        tmp = df.copy()
        tmp.columns = [col_map.get(str(c), str(c)) for c in tmp.columns]
        for c in ["ts", "open", "high", "low", "close"]:
            if c not in tmp.columns:
                raise RuntimeError(f"missing_column_{c}")
        if "volume" not in tmp.columns:
            tmp["volume"] = 0.0
        if "amount" not in tmp.columns:
            tmp["amount"] = 0.0
        if "pct_chg" not in tmp.columns:
            tmp["pct_chg"] = 0.0
        out = tmp[["ts", "open", "high", "low", "close", "volume", "amount", "pct_chg"]].copy()
        for c in ["open", "high", "low", "close", "volume", "amount", "pct_chg"]:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        out["ts"] = out["ts"].astype(str)
        return out.sort_values("ts").reset_index(drop=True)

    def _fetch_fundamentals(self, symbol: str) -> dict[str, Any]:
        data: dict[str, Any] = {"source": "unknown"}
        # 优先 tushare
        if self._tushare_token:
            try:
                _tushare_throttle()
                import tushare as ts

                ts.set_token(self._tushare_token)
                pro = ts.pro_api(timeout=self._tushare_timeout)
                ts_code = self._to_ts_code(symbol)
                basic = pro.daily_basic(ts_code=ts_code, limit=1)
                if not basic.empty:
                    row = basic.iloc[0]
                    data = {
                        "source": "tushare",
                        "pe_ttm": self._safe_float(row.get("pe_ttm")),
                        "pb": self._safe_float(row.get("pb")),
                        "turnover_rate": self._safe_float(row.get("turnover_rate")),
                        "total_mv": self._safe_float(row.get("total_mv")),
                    }
                    return data
            except Exception:
                pass
        # 兜底 akshare
        try:
            import akshare as ak

            info = ak.stock_individual_info_em(symbol=symbol)
            kv = {}
            for _, r in info.iterrows():
                k = str(r.iloc[0]).strip()
                v = str(r.iloc[1]).strip()
                kv[k] = v
            data = {
                "source": "akshare",
                "pe_ttm": self._safe_float(kv.get("市盈率-动态") or kv.get("市盈率动态")),
                "pb": self._safe_float(kv.get("市净率")),
                "total_mv": self._safe_float(kv.get("总市值")),
            }
        except Exception:
            data = {"source": "unknown"}
        # 再兜底：从A股行情快照提取常见估值字段
        if data.get("source") == "unknown":
            try:
                import akshare as ak

                spot = ak.stock_zh_a_spot_em()
                row = spot[spot["代码"] == symbol]
                if not row.empty:
                    r = row.iloc[0]
                    data = {
                        "source": "akshare_spot",
                        "pe_ttm": self._safe_float(r.get("市盈率-动态")),
                        "pb": self._safe_float(r.get("市净率")),
                        "turnover_rate": self._safe_float(r.get("换手率")),
                        "total_mv": self._safe_float(r.get("总市值")),
                    }
            except Exception:
                pass
        return data

    def _fetch_news(self, symbol: str) -> list[dict[str, Any]]:
        try:
            import akshare as ak

            df = ak.stock_news_em(symbol=symbol)
            out = []
            for _, r in df.head(80).iterrows():
                out.append(
                    {
                        "title": str(r.get("新闻标题", "")),
                        "content": str(r.get("新闻内容", ""))[:300],
                        "time": str(r.get("发布时间", "")),
                        "source": str(r.get("文章来源", "")),
                        "url": str(r.get("新闻链接", "")),
                    }
                )
            return out
        except Exception:
            return []

    def _compute_features(
        self,
        snap: MarketSnapshot,
        hist_df,
        fundamentals: dict[str, Any],
        news: list[dict[str, Any]],
        kline_bundle: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        features: dict[str, Any] = {
            "pct_chg": snap.pct_chg,
            "close": snap.close,
            "pe_ttm": snap.pe_ttm if snap.pe_ttm is not None else fundamentals.get("pe_ttm"),
            "turnover_rate": snap.turnover_rate if snap.turnover_rate is not None else fundamentals.get("turnover_rate"),
            "pb": fundamentals.get("pb"),
            "total_mv": fundamentals.get("total_mv"),
            "momentum_20": 0.0,
            "volatility_20": 0.0,
            "drawdown_60": 0.0,
            "volume_ratio_5_20": 1.0,
            "trend_strength": 0.0,
            "news_sentiment": 0.0,
            "news_count": len(news),
            "data_quality_score": 0.0,
            "_has_hist_data": bool(hist_df is not None and not hist_df.empty),
            "_has_fundamental_data": fundamentals.get("source") in {"tushare", "akshare", "akshare_spot"},
            "_has_news_data": len(news) > 0,
            "kline_indicators": {},
        }
        if hist_df is not None and not hist_df.empty:
            try:
                close_col = "close" if "close" in hist_df.columns else "收盘"
                vol_col = "volume" if "volume" in hist_df.columns else "成交量"
                close = hist_df[close_col].astype(float)
                ret = close.pct_change()
                if len(close) >= 21:
                    features["momentum_20"] = float((close.iloc[-1] / close.iloc[-21] - 1.0) * 100)
                    features["volatility_20"] = float(ret.tail(20).std() * 100)
                if len(close) >= 60:
                    roll_max = close.tail(60).cummax()
                    dd = (close.tail(60) / roll_max - 1.0) * 100
                    features["drawdown_60"] = float(dd.min())
                if len(close) >= 20:
                    features["trend_strength"] = float((close.tail(20).iloc[-1] - close.tail(20).iloc[0]) / close.tail(20).mean() * 100)
                if vol_col in hist_df.columns and len(hist_df) >= 20:
                    vol = hist_df[vol_col].astype(float)
                    v5 = vol.tail(5).mean()
                    v20 = vol.tail(20).mean()
                    features["volume_ratio_5_20"] = float(v5 / v20) if v20 else 1.0
            except Exception:
                pass

        features["news_sentiment"] = self._calc_news_sentiment(news)
        if kline_bundle:
            features["kline_indicators"] = self._compute_multi_kline_indicators(kline_bundle)
            features["_has_hist_data"] = all(v.get("ok") for v in kline_bundle.values())
            day_item = kline_bundle.get("day", {})
            day_df = day_item.get("df") if isinstance(day_item, dict) else None
            if day_df is not None and not day_df.empty and snap.close:
                features["key_levels"] = self._compute_fibonacci_key_levels(day_df, float(snap.close))
            else:
                features["key_levels"] = {}
        else:
            features["key_levels"] = {}
        # 数据质量：按关键字段完整度
        checks = [
            features.get("_has_hist_data", False),
            features.get("_has_fundamental_data", False),
            features.get("_has_news_data", False),
            features.get("pe_ttm") is not None,
            features.get("turnover_rate") is not None,
            features.get("momentum_20") != 0.0,
            features.get("volatility_20") != 0.0,
        ]
        valid = sum(1 for c in checks if c)
        features["data_quality_score"] = round(valid / len(checks), 4)
        return features

    def _compute_multi_kline_indicators(self, kline_bundle: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """仅基于日线/周线/月线计算技术指标。含MACD、RSI、KDJ、布林带、StochRSI、波动率、3-5根K线组合、长期趋势线。"""
        import pandas as pd

        # 仅处理日/周/月
        allowed_tfs = {"day", "week", "month"}
        out: dict[str, Any] = {}
        for tf, item in kline_bundle.items():
            if tf not in allowed_tfs:
                continue
            df = item.get("df")
            if df is None or df.empty:
                out[tf] = {"ok": False, "rows": 0, "timeframe_label": {"day": "日线", "week": "周线", "month": "月线"}.get(tf, tf)}
                continue
            close = df["close"].astype(float)
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            open_ = df["open"].astype(float) if "open" in df.columns else close
            ret = close.pct_change().fillna(0.0)
            n = len(close)

            mom = float((close.iloc[-1] / close.iloc[-11] - 1.0) * 100) if n >= 11 else 0.0
            vol = float(ret.tail(20).std() * 100) if n >= 20 else 0.0
            amp = float(((high.tail(20) - low.tail(20)) / close.tail(20).replace(0, 1)).mean() * 100) if n >= 20 else 0.0

            # 上下影线
            last5 = df.tail(5)
            if len(last5) >= 3:
                h5 = last5["high"].astype(float)
                l5 = last5["low"].astype(float)
                o5 = last5["open"].astype(float) if "open" in last5.columns else last5["close"].astype(float)
                c5 = last5["close"].astype(float)
                hl = (h5 - l5).replace(0, 1e-8)
                body_top = pd.concat([o5, c5], axis=1).max(axis=1)
                body_bottom = pd.concat([o5, c5], axis=1).min(axis=1)
                upper_shadow = float(((h5 - body_top) / hl).mean() * 100)
                lower_shadow = float(((body_bottom - l5) / hl).mean() * 100)
            else:
                upper_shadow, lower_shadow = 0.0, 0.0

            # RSI(14)
            rsi_val = self._calc_rsi(close, 14) if n >= 15 else None
            # MACD(12,26,9)
            macd_val = self._calc_macd(close) if n >= 35 else None
            # KDJ(9,3,3)
            kdj_val = self._calc_kdj(high, low, close, 9, 3, 3) if n >= 11 else None
            # 布林带(20,2)
            boll_val = self._calc_bollinger(close, 20, 2) if n >= 22 else None
            # StochRSI(14)
            stoch_rsi_val = self._calc_stoch_rsi(close, 14) if n >= 29 else None
            # 3-5根K线组合（三连阳/三连阴等）
            k_combo = self._detect_kline_combo(open_, high, low, close, 5) if n >= 5 else None
            # 长期趋势线（20周期线性回归斜率%/根）
            trend_slope = self._calc_trend_slope(close, 20) if n >= 20 else None

            # 均线系统 MA5/MA10/MA20/MA60/MA120/MA250
            curr_price = float(close.iloc[-1])
            ma_system: dict[str, dict] = {}
            for period in [5, 10, 20, 60, 120, 250]:
                if n >= period:
                    ma_val = float(close.rolling(period).mean().iloc[-1])
                    pct_above = (curr_price - ma_val) / ma_val * 100 if ma_val > 0 else 0.0
                    ma_system[f"ma{period}"] = {
                        "value": round(ma_val, 4),
                        "pct_above": round(pct_above, 2),
                    }
                else:
                    ma_system[f"ma{period}"] = {"value": None, "pct_above": None}

            out[tf] = {
                "ok": True,
                "rows": int(n),
                "timeframe_label": {"day": "日线", "week": "周线", "month": "月线"}.get(tf, tf),
                "momentum_10": round(mom, 4),
                "volatility_20": round(vol, 4),
                "amplitude_20": round(amp, 4),
                "upper_shadow_ratio": round(upper_shadow, 4),
                "lower_shadow_ratio": round(lower_shadow, 4),
                "rsi": round(rsi_val, 2) if rsi_val is not None else None,
                "macd_dif": round(macd_val[0], 4) if macd_val else None,
                "macd_dea": round(macd_val[1], 4) if macd_val and len(macd_val) > 1 else None,
                "macd_hist": round(macd_val[2], 4) if macd_val and len(macd_val) > 2 else None,
                "kdj_k": round(kdj_val[0], 2) if kdj_val else None,
                "kdj_d": round(kdj_val[1], 2) if kdj_val and len(kdj_val) > 1 else None,
                "kdj_j": round(kdj_val[2], 2) if kdj_val and len(kdj_val) > 2 else None,
                "boll_upper": round(boll_val[0], 4) if boll_val else None,
                "boll_mid": round(boll_val[1], 4) if boll_val and len(boll_val) > 1 else None,
                "boll_lower": round(boll_val[2], 4) if boll_val and len(boll_val) > 2 else None,
                "stoch_rsi": round(stoch_rsi_val, 2) if stoch_rsi_val is not None else None,
                "kline_combo_5": k_combo,
                "trend_slope_pct": round(trend_slope, 4) if trend_slope is not None else None,
                "ma_system": ma_system,
                "close": round(curr_price, 4),
            }
        return out

    @staticmethod
    def _calc_rsi(close, period: int = 14) -> float | None:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if len(rsi.dropna()) else None

    @staticmethod
    def _calc_macd(close, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float, float, float] | None:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal, adjust=False).mean()
        hist = dif - dea
        if len(dif.dropna()) < signal:
            return None
        return float(dif.iloc[-1]), float(dea.iloc[-1]), float(hist.iloc[-1])

    @staticmethod
    def _calc_kdj(high, low, close, n: int = 9, m1: int = 3, m2: int = 3) -> tuple[float, float, float] | None:
        lowest = low.rolling(n).min()
        highest = high.rolling(n).max()
        rsv = (close - lowest) / (highest - lowest).replace(0, 1e-10) * 100
        k = rsv.ewm(alpha=1 / m1, adjust=False).mean()
        d = k.ewm(alpha=1 / m2, adjust=False).mean()
        j = 3 * k - 2 * d
        if len(k.dropna()) == 0:
            return None
        return float(k.iloc[-1]), float(d.iloc[-1]), float(j.iloc[-1])

    @staticmethod
    def _calc_bollinger(close, period: int = 20, std_mult: float = 2.0) -> tuple[float, float, float] | None:
        mid = close.rolling(period).mean()
        std = close.rolling(period).std().fillna(0)
        upper = mid + std_mult * std
        lower = mid - std_mult * std
        if len(mid.dropna()) == 0:
            return None
        return float(upper.iloc[-1]), float(mid.iloc[-1]), float(lower.iloc[-1])

    @staticmethod
    def _calc_stoch_rsi(close, period: int = 14) -> float | None:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean().replace(0, 1e-10)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi_min = rsi.rolling(14).min()
        rsi_max = rsi.rolling(14).max()
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, 1e-10) * 100
        if len(stoch_rsi.dropna()) == 0:
            return None
        return float(stoch_rsi.iloc[-1])

    @staticmethod
    def _detect_kline_combo(open_, high, low, close, lookback: int = 5) -> str | None:
        """检测3-5根K线组合形态。"""
        if len(close) < lookback:
            return None
        last = lookback
        o = open_.tail(last).values
        h = high.tail(last).values
        l_ = low.tail(last).values
        c = close.tail(last).values
        body = c - o
        # 三连阳
        if last >= 3 and all(body[-3:] > 0):
            return "三连阳"
        # 三连阴
        if last >= 3 and all(body[-3:] < 0):
            return "三连阴"
        # 两阳夹一阴
        if last >= 3 and body[-3] > 0 and body[-2] < 0 and body[-1] > 0:
            return "两阳夹一阴"
        # 早晨之星：阴-十字/小实体-阳
        if last >= 3 and body[-3] < -0.001 and abs(body[-2]) < 0.002 and body[-1] > 0.005:
            return "早晨之星"
        return "无典型组合"

    @staticmethod
    def _compute_fibonacci_key_levels(day_df, current_close: float) -> dict[str, Any]:
        """基于日线计算近期波段高低点及 Fibonacci 回撤位（23.6%/38.2%/50%/61.8%）。"""
        import pandas as pd

        if day_df is None or day_df.empty or len(day_df) < 20:
            return {}
        try:
            high = day_df["high"].astype(float)
            low = day_df["low"].astype(float)
            close = day_df["close"].astype(float)
            n = len(close)
            lookback = min(60, n - 1)
            seg_high = high.tail(lookback).max()
            seg_low = low.tail(lookback).min()
            idx_high = high.tail(lookback).idxmax()
            idx_low = low.tail(lookback).idxmin()
            # 确定上涨波段(低→高)或下跌波段(高→低)：近期谁更晚
            try:
                pos_high = day_df.index.get_loc(idx_high) if hasattr(day_df.index, 'get_loc') else list(day_df.index).index(idx_high)
                pos_low = day_df.index.get_loc(idx_low) if hasattr(day_df.index, 'get_loc') else list(day_df.index).index(idx_low)
            except Exception:
                pos_high, pos_low = 0, 0
            if pos_high > pos_low:
                # 先低后高：上涨波段，回撤从高到低
                band_high, band_low = seg_high, seg_low
            else:
                # 先高后低：下跌波段，反弹从低到高
                band_high, band_low = seg_low, seg_high
            diff = band_high - band_low
            if diff <= 0:
                return {}
            r236 = band_high - 0.236 * diff
            r382 = band_high - 0.382 * diff
            r50 = band_high - 0.5 * diff
            r618 = band_high - 0.618 * diff
            return {
                "ok": True,
                "band_high": round(band_high, 2),
                "band_low": round(band_low, 2),
                "current": round(current_close, 2),
                "retrace_236": round(r236, 2),
                "retrace_382": round(r382, 2),
                "retrace_50": round(r50, 2),
                "retrace_618": round(r618, 2),
            }
        except Exception:
            return {}

    @staticmethod
    def _calc_trend_slope(close, period: int = 20) -> float | None:
        """线性回归斜率（% per bar），正值上行、负值下行。"""
        import numpy as np

        tail = close.tail(period)
        x = np.arange(len(tail))
        y = tail.values
        if len(x) < 2:
            return None
        slope = np.polyfit(x, y, 1)[0]
        mean_price = float(y.mean())
        return (slope / mean_price * 100) if mean_price else None

    @staticmethod
    def _snapshot_from_klines(symbol: str, name: str, day_df) -> MarketSnapshot | None:
        if day_df is None or day_df.empty:
            return None
        try:
            last = day_df.iloc[-1]
            close = float(last.get("close", 0.0) or 0.0)
            pct = float(last.get("pct_chg", 0.0) or 0.0)
            return MarketSnapshot(
                symbol=symbol,
                name=name,
                close=close,
                pct_chg=pct,
                pe_ttm=None,
                turnover_rate=None,
                source="kline_day_fallback",
            )
        except Exception:
            return None

    @staticmethod
    def _calc_news_sentiment(news: list[dict[str, Any]]) -> float:
        if not news:
            return 0.0
        pos_words = ["增长", "中标", "利好", "回购", "增持", "突破", "新高", "盈利"]
        neg_words = ["下滑", "减持", "问询", "处罚", "亏损", "风险", "诉讼", "违约"]
        pos = 0
        neg = 0
        for n in news[:80]:
            txt = f"{n.get('title','')} {n.get('content','')}"
            pos += sum(1 for w in pos_words if w in txt)
            neg += sum(1 for w in neg_words if w in txt)
        total = pos + neg
        if total == 0:
            return 0.0
        return round((pos - neg) / total * 100, 4)

    @staticmethod
    def _safe_float(v: Any) -> float | None:
        if v is None:
            return None
        try:
            s = str(v).replace(",", "").replace("%", "").strip()
            if s in {"", "-", "--", "None", "nan"}:
                return None
            return float(s)
        except Exception:
            return None

    def _fetch_akshare(self, symbol: str, name: str) -> MarketSnapshot:
        df = self._fetch_hist_akshare(symbol)
        if df is not None and not df.empty:
            last = df.iloc[-1]
            close = float(last["收盘"])
            pct = float(last["涨跌幅"])
            return MarketSnapshot(
                symbol=symbol,
                name=name,
                close=close,
                pct_chg=pct,
                pe_ttm=None,
                turnover_rate=None,
                source="akshare",
            )
        # 兜底：实时行情
        try:
            import akshare as ak

            spot = ak.stock_zh_a_spot_em()
            row = spot[spot["代码"] == symbol]
            if not row.empty:
                r = row.iloc[0]
                return MarketSnapshot(
                    symbol=symbol,
                    name=str(r.get("名称") or name),
                    close=float(r.get("最新价") or 0.0),
                    pct_chg=float(r.get("涨跌幅") or 0.0),
                    pe_ttm=self._safe_float(r.get("市盈率-动态")),
                    turnover_rate=self._safe_float(r.get("换手率")),
                    source="akshare_spot",
                )
        except Exception:
            pass
        raise RuntimeError("akshare snapshot unavailable")

    def _fetch_tushare(self, symbol: str, name: str) -> MarketSnapshot:
        import tushare as ts

        if not self._tushare_token:
            raise RuntimeError("missing TUSHARE_TOKEN")
        _tushare_throttle()
        ts.set_token(self._tushare_token)
        pro = ts.pro_api(timeout=self._tushare_timeout)
        ts_code = self._to_ts_code(symbol)
        daily = pro.daily(ts_code=ts_code, limit=1)
        _tushare_throttle()
        basic = pro.daily_basic(ts_code=ts_code, limit=1)
        close = float(daily.iloc[0]["close"])
        pct = float(daily.iloc[0]["pct_chg"])
        pe_ttm = float(basic.iloc[0]["pe_ttm"]) if "pe_ttm" in basic.columns else None
        turnover = float(basic.iloc[0]["turnover_rate"]) if "turnover_rate" in basic.columns else None
        return MarketSnapshot(
            symbol=symbol,
            name=name,
            close=close,
            pct_chg=pct,
            pe_ttm=pe_ttm,
            turnover_rate=turnover,
            source="tushare",
        )

    @staticmethod
    def _clean_symbol(symbol: str) -> str:
        """标准化股票代码为纯6位数字（去除 .SZ/.SH 等后缀和 sz/sh 等前缀）。"""
        s = symbol.strip().upper()
        for suffix in (".SZ", ".SH", ".BJ"):
            if s.endswith(suffix):
                s = s[: -len(suffix)]
                break
        if len(s) > 6 and s[:2] in ("SZ", "SH", "BJ"):
            s = s[2:]
        return s

    @staticmethod
    def _to_ts_code(symbol: str) -> str:
        s = DataBackend._clean_symbol(symbol)
        if s.startswith("6"):
            return f"{s}.SH"
        if s.startswith("8") or s.startswith("4"):
            return f"{s}.BJ"
        return f"{s}.SZ"
