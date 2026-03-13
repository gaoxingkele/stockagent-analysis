# -*- coding: utf-8 -*-
import os
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv


load_dotenv()

# 国内数据源全部直连，不走系统代理（海外 LLM 代理会阻断国内站点）。
# 覆盖：Tushare、东财(AKShare)、腾讯财经、新浪财经、同花顺 等。
# 如需追加，可设 DATA_NO_PROXY 环境变量（逗号分隔）。
_DATA_NO_PROXY = (
    # Tushare
    "api.tushare.pro,api.waditu.com,"
    # 东方财富 (AKShare 主力源)
    ".eastmoney.com,push2his.eastmoney.com,push2.eastmoney.com,"
    "datacenter-web.eastmoney.com,data.eastmoney.com,"
    # 腾讯财经
    ".gtimg.cn,stock.gtimg.cn,web.ifzq.gtimg.cn,qt.gtimg.cn,"
    "web.sqt.gtimg.cn,proxy.finance.qq.com,"
    # 新浪财经
    ".sinajs.cn,.sina.com.cn,hq.sinajs.cn,"
    "vip.stock.finance.sina.com.cn,"
    # 同花顺 / 其他国内站点
    ".10jqka.com.cn,.mairui.club,.akshare.akfamily.xyz"
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
    """确保 Tushare 域名在 NO_PROXY 中，数据采集不走 LLM 代理。
    可通过 DATA_NO_PROXY 环境变量追加额外需要绕过的域名（逗号分隔）。
    """
    extra = os.getenv("DATA_NO_PROXY", "").strip()
    domains = _DATA_NO_PROXY + ("," + extra if extra else "")
    current = os.environ.get("NO_PROXY", "").strip()
    if not current:
        os.environ["NO_PROXY"] = domains
    else:
        existing = {d.strip() for d in current.split(",") if d.strip()}
        to_add = [d.strip() for d in domains.split(",") if d.strip() and d.strip() not in existing]
        if to_add:
            os.environ["NO_PROXY"] = f"{current},{','.join(to_add)}".strip(",")
    os.environ["no_proxy"] = os.environ["NO_PROXY"]


_INDUSTRY_ETF_MAP: dict[str, str] = {
    "银行": "515290", "证券": "512880", "保险": "512070",
    "房地产": "512200", "医药生物": "512010", "食品饮料": "515170",
    "家用电器": "159996", "汽车": "516110", "电力设备": "516160",
    "机械设备": "516960", "电子": "515260", "半导体": "512480",
    "计算机": "512720", "通信": "515880", "传媒": "516880",
    "国防军工": "512660", "钢铁": "515210", "有色金属": "512400",
    "化工": "516220", "煤炭": "515220", "石油石化": "516060",
    "电力": "159611", "农林牧渔": "516670", "光伏": "515790",
    "白酒": "512690", "中药": "560080", "创新药": "516260",
    "芯片": "512480", "游戏": "516010", "建筑材料": "516750",
    "医疗器械": "159883", "新能源汽车": "515030", "稀土": "516150",
    "航空航天": "515850",
}


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
        self._tdx_reader = None  # lazy init; None means not yet tried
        self._tdx_reader_ok: bool | None = None  # True/False after first try
        _ensure_akshare_no_proxy()

    def _get_tdx_reader(self):
        """获取通达信本地数据读取器（惰性初始化，TDX目录不存在时返回None）。"""
        if self._tdx_reader_ok is False:
            return None
        if self._tdx_reader is not None:
            return self._tdx_reader
        try:
            from mootdx.reader import Reader
            tdx_dir = os.getenv("TDX_DIR", "D:/tdx")
            vipdoc = os.path.join(tdx_dir, "vipdoc")
            if not os.path.isdir(vipdoc):
                self._tdx_reader_ok = False
                return None
            self._tdx_reader = Reader.factory(market="std", tdxdir=tdx_dir)
            self._tdx_reader_ok = True
            return self._tdx_reader
        except Exception:
            self._tdx_reader_ok = False
            return None

    def fetch_snapshot(self, symbol: str, name: str, preferred_sources: list[str] | None = None) -> MarketSnapshot:
        symbol = self._clean_symbol(symbol)
        sources = list(preferred_sources or self.default_sources)
        # 通达信本地数据最高优先级（无网络，速度最快）
        if "tdx" not in sources:
            sources = ["tdx"] + sources
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
        # 融资融券 + 北向资金（Tushare权限不足时返回空dict，不影响主流程）
        features["margin_data"] = self._fetch_margin_data(symbol)
        features["hsgt_data"] = self._fetch_hsgt_data(symbol)
        # 市场状态识别
        features["market_regime"] = self._detect_market_regime()
        # 市场策略框架（三阶段映射）
        from .market_strategy import determine_strategy, strategy_to_dict
        features["market_strategy"] = strategy_to_dict(determine_strategy(features["market_regime"]))
        # 筹码结构分析（基于日线量价模拟）
        if day_df is not None and not day_df.empty and snapshot.close:
            features["chip_distribution"] = self._fetch_chip_distribution(day_df, float(snapshot.close))
        else:
            features["chip_distribution"] = {}
        # 相对强弱分析（个股 vs 沪深300）
        features["relative_strength"] = self._fetch_and_compute_rs(kline_bundle)
        # 相对强弱分析（个股 vs 行业板块指数）
        features["rs_vs_industry"] = self._fetch_and_compute_rs_industry(kline_bundle, fundamentals)
        # 相对强弱分析（个股 vs 板块龙头TOP3）
        features["rs_vs_leaders"] = self._fetch_and_compute_rs_leaders(kline_bundle, fundamentals, symbol)
        # 相对强弱分析（个股 vs 行业ETF）
        features["rs_vs_etf"] = self._fetch_and_compute_rs_etf(kline_bundle, fundamentals)
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

        required_tfs = ["day", "week"]  # month 为可选，缺失时降级而非终止
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
        """K线抓取：通达信本地 → Tushare → AKShare，依次降级。每周期最多重试 3 次。"""
        # 确保顺序：通达信本地最优先，其次 Tushare，最后 AKShare
        ordered_sources = ["tdx", "akshare", "tushare"]
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
                        if source == "tdx":
                            data = self._fetch_kline_tdx(symbol, tf, limit)
                        elif source == "akshare":
                            data = self._fetch_kline_akshare(symbol, tf, limit)
                        elif source == "tushare":
                            data = self._fetch_kline_tushare(symbol, tf, limit)
                        else:
                            raise ValueError(f"unsupported source: {source}")
                        # 新股K线少，有多少拿多少，只要非空即可
                        if data is not None and not data.empty and len(data) >= 1:
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
        if source == "tdx":
            return self._fetch_snapshot_tdx(symbol, name)
        if source == "akshare":
            return self._fetch_akshare(symbol, name)
        if source == "tushare":
            return self._fetch_tushare(symbol, name)
        raise ValueError(f"unsupported source: {source}")

    def _fetch_kline_tdx(self, symbol: str, timeframe: str, limit: int):
        """从通达信本地文件读取K线（无网络请求，速度最快）。
        日/周/月：从 vipdoc/*/lday/*.day 读取后按需重采样。
        1h：从 vipdoc/*/minline/*.lc1 读取1分钟数据后重采样至60分钟。
        """
        import pandas as pd

        reader = self._get_tdx_reader()
        if reader is None:
            raise RuntimeError("tdx_reader_unavailable")

        if timeframe in ("day", "week", "month"):
            df = reader.daily(symbol=symbol)
            if df is None or df.empty:
                raise RuntimeError(f"tdx_{timeframe}_no_data")

            # ── TDX数据不是最新时，用AKShare补足缺失交易日 ──
            df = self._patch_tdx_with_akshare(df, symbol)
            # ── 盘中分析：用1h K线合成当天虚拟日线 ──
            df = self._patch_intraday_as_daily(df, symbol)

            if timeframe == "week":
                df = df.resample("W-FRI").agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last", "amount": "sum", "volume": "sum"}
                ).dropna()
            elif timeframe == "month":
                df = df.resample("ME").agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last", "amount": "sum", "volume": "sum"}
                ).dropna()

            df = df.reset_index()
            date_col = df.columns[0]  # "date" after reset_index
            df.rename(columns={date_col: "ts"}, inplace=True)
            df["ts"] = pd.to_datetime(df["ts"]).dt.strftime("%Y-%m-%d")
            df["pct_chg"] = df["close"].pct_change() * 100
            df["pct_chg"] = df["pct_chg"].fillna(0.0)

        elif timeframe == "1h":
            df_min = reader.minute(symbol=symbol, suffix=1)
            if df_min is None or df_min.empty:
                raise RuntimeError("tdx_1h_no_minute_data")

            df = df_min.resample("1h").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "amount": "sum", "volume": "sum"}
            ).dropna()
            df = df.reset_index()
            dt_col = df.columns[0]
            df.rename(columns={dt_col: "ts"}, inplace=True)
            df["ts"] = pd.to_datetime(df["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            df["pct_chg"] = df["close"].pct_change() * 100
            df["pct_chg"] = df["pct_chg"].fillna(0.0)

        else:
            raise RuntimeError(f"tdx_unsupported_timeframe_{timeframe}")

        result = df[["ts", "open", "high", "low", "close", "volume", "amount", "pct_chg"]].copy()
        for c in ["open", "high", "low", "close", "volume", "amount", "pct_chg"]:
            result[c] = pd.to_numeric(result[c], errors="coerce").fillna(0.0)
        result = result.sort_values("ts").reset_index(drop=True)

        if len(result) < 1:
            raise RuntimeError(f"tdx_{timeframe}_rows_not_enough (got {len(result)})")

        return result.tail(limit).reset_index(drop=True)

    def _patch_tdx_with_akshare(self, df, symbol: str):
        """TDX日线数据不含今日时，用AKShare补足缺失的交易日。

        df: TDX读取的日线DataFrame，index=DatetimeIndex(name='date')。
        返回补足后的DataFrame（同格式）。
        """
        import pandas as pd

        today = pd.Timestamp(datetime.now().date())
        tdx_last = df.index[-1]
        if tdx_last >= today:
            return df  # 已是最新，无需补足

        # 用AKShare获取TDX最后日期之后的数据
        try:
            import akshare as ak

            start = (tdx_last + timedelta(days=1)).strftime("%Y%m%d")
            end = today.strftime("%Y%m%d")
            patch = ak.stock_zh_a_hist(
                symbol=symbol, period="daily",
                start_date=start, end_date=end, adjust="",
            )
            if patch is None or patch.empty:
                return df

            # 标准化列名以匹配TDX格式
            col_map = {"日期": "date", "开盘": "open", "最高": "high",
                        "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount"}
            patch = patch.rename(columns=col_map)
            for c in ["open", "high", "low", "close", "volume", "amount"]:
                if c in patch.columns:
                    patch[c] = pd.to_numeric(patch[c], errors="coerce").fillna(0.0)
            patch["date"] = pd.to_datetime(patch["date"])
            patch = patch.set_index("date")
            # 只保留TDX已有的列
            common_cols = [c for c in df.columns if c in patch.columns]
            patch = patch[common_cols]

            merged = pd.concat([df, patch])
            merged = merged[~merged.index.duplicated(keep="last")]
            merged = merged.sort_index()
            n_patched = len(merged) - len(df)
            if n_patched > 0:
                print(f"[数据] TDX补足 {symbol}: +{n_patched}根日线 (AKShare {start}~{end})", flush=True)
            return merged
        except Exception:
            return df  # 补足失败，静默回退

    def _patch_intraday_as_daily(self, df, symbol: str):
        """盘中分析：用当天1小时K线合成虚拟日线，追加到日线数据末尾。

        适用场景：盘中（如午盘后）运行分析，当天日线尚未收盘。
        从AKShare获取当天已走完的1小时K线，合成OHLCV作为当天的"虚拟日线"。
        如果当天日线已存在（收盘后运行），则不做任何操作。
        """
        import pandas as pd

        today = pd.Timestamp(datetime.now().date())
        if df.index[-1] >= today:
            return df  # 今天的日线已存在，无需合成

        # 判断当前是否在交易时间段（9:15~15:05）
        now = datetime.now()
        if now.weekday() >= 5:
            return df  # 周末不合成
        from datetime import time as dt_time
        if now.time() < dt_time(9, 30):
            return df  # 开盘前不合成

        try:
            import akshare as ak

            start_dt = today.strftime("%Y-%m-%d 09:30:00")
            end_dt = now.strftime("%Y-%m-%d %H:%M:%S")
            raw = ak.stock_zh_a_hist_min_em(
                symbol=symbol, period="60",
                start_date=start_dt, end_date=end_dt, adjust="",
            )
            if raw is None or raw.empty:
                return df

            # 标准化列名
            col_map = {"时间": "ts", "开盘": "open", "最高": "high",
                       "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount"}
            raw = raw.rename(columns=col_map)
            for c in ["open", "high", "low", "close", "volume", "amount"]:
                if c in raw.columns:
                    raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0.0)

            # 只取今天的1h K线
            if "ts" in raw.columns:
                raw["ts"] = pd.to_datetime(raw["ts"])
                raw = raw[raw["ts"].dt.date == today.date()]

            if raw.empty:
                return df

            # 合成虚拟日线：open取第一根，high取最高，low取最低，close取最后一根
            virtual_daily = pd.DataFrame([{
                "open": float(raw.iloc[0]["open"]),
                "high": float(raw["high"].max()),
                "low": float(raw["low"].min()),
                "close": float(raw.iloc[-1]["close"]),
                "volume": float(raw["volume"].sum()),
                "amount": float(raw["amount"].sum()),
            }], index=pd.DatetimeIndex([today], name="date"))

            # 只保留df已有的列
            common_cols = [c for c in df.columns if c in virtual_daily.columns]
            virtual_daily = virtual_daily[common_cols]

            merged = pd.concat([df, virtual_daily])
            merged = merged[~merged.index.duplicated(keep="last")]
            merged = merged.sort_index()

            n_bars = len(raw)
            print(f"[数据] 盘中合成日线 {symbol}: {n_bars}根1h线 → 虚拟日线 "
                  f"O={virtual_daily.iloc[0]['open']:.2f} H={virtual_daily.iloc[0]['high']:.2f} "
                  f"L={virtual_daily.iloc[0]['low']:.2f} C={virtual_daily.iloc[0]['close']:.2f}",
                  flush=True)
            return merged
        except Exception:
            return df  # 合成失败，静默回退

    def _fetch_snapshot_tdx(self, symbol: str, name: str) -> MarketSnapshot:
        """从通达信本地日线文件读取最新行情快照（取最后一根K线）。"""
        reader = self._get_tdx_reader()
        if reader is None:
            raise RuntimeError("tdx_reader_unavailable")
        df = reader.daily(symbol=symbol)
        if df is None or df.empty:
            raise RuntimeError("tdx_snapshot_no_data")
        close = float(df.iloc[-1]["close"])
        pct_chg = 0.0
        if len(df) >= 2:
            prev = float(df.iloc[-2]["close"])
            if prev > 0:
                pct_chg = (close / prev - 1.0) * 100
        return MarketSnapshot(
            symbol=symbol,
            name=name,
            close=close,
            pct_chg=round(pct_chg, 4),
            pe_ttm=None,
            turnover_rate=None,
            source="tdx",
        )

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
            # 日线取1年，周线取3年，月线取10年，确保足够的K线根数
            lookback_days = {"day": 365, "week": 1095, "month": 3650}.get(timeframe, 365)
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y%m%d")
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
        if len(df) < 1:
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
                # 追加财务指标（ROE/毛利率/增速等）
                try:
                    _tushare_throttle()
                    fina = pro.fina_indicator(ts_code=ts_code, limit=1)
                    if fina is not None and not fina.empty:
                        row_f = fina.iloc[0]
                        data["roe"] = self._safe_float(row_f.get("roe"))
                        data["grossprofit_margin"] = self._safe_float(row_f.get("grossprofit_margin"))
                        data["netprofit_margin"] = self._safe_float(row_f.get("netprofit_margin"))
                        data["debt_to_assets"] = self._safe_float(row_f.get("debt_to_assets"))
                        data["revenue_yoy"] = self._safe_float(row_f.get("tr_yoy"))
                        data["netprofit_yoy"] = self._safe_float(row_f.get("q_profit_yoy"))
                        data["eps"] = self._safe_float(row_f.get("eps"))
                        data["cfps"] = self._safe_float(row_f.get("cfps"))
                except Exception:
                    pass
                # 获取行业名称（用于RS vs行业分析）
                if not data.get("industry"):
                    try:
                        import akshare as ak
                        info = ak.stock_individual_info_em(symbol=symbol)
                        for _, r in info.iterrows():
                            if str(r.iloc[0]).strip() == "行业":
                                data["industry"] = str(r.iloc[1]).strip()
                                break
                    except Exception:
                        pass
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
                "industry": kv.get("行业", ""),
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

    def _fetch_margin_data(self, symbol: str) -> dict[str, Any]:
        """获取融资融券数据（最近20个交易日）。"""
        if not self._tushare_token:
            return {}
        try:
            import tushare as ts
            _tushare_throttle()
            ts.set_token(self._tushare_token)
            pro = ts.pro_api(timeout=self._tushare_timeout)
            ts_code = self._to_ts_code(symbol)
            margin = pro.margin_detail(ts_code=ts_code, limit=20)
            if margin is None or margin.empty:
                return {}
            latest = margin.iloc[0]
            return {
                "rzye": self._safe_float(latest.get("rzye")),
                "rzmre": self._safe_float(latest.get("rzmre")),
                "rqye": self._safe_float(latest.get("rqye")),
                "rqmcl": self._safe_float(latest.get("rqmcl")),
                "rzye_change_5d": self._calc_series_change(margin, "rzye", 5),
                "rzye_change_10d": self._calc_series_change(margin, "rzye", 10),
                "source": "tushare",
            }
        except Exception:
            return {}

    def _fetch_hsgt_data(self, symbol: str) -> dict[str, Any]:
        """获取北向资金（沪深港通）持股数据。"""
        if not self._tushare_token:
            return {}
        try:
            import tushare as ts
            _tushare_throttle()
            ts.set_token(self._tushare_token)
            pro = ts.pro_api(timeout=self._tushare_timeout)
            ts_code = self._to_ts_code(symbol)
            hsgt = pro.hk_hold(ts_code=ts_code, limit=20)
            if hsgt is None or hsgt.empty:
                return {}
            latest = hsgt.iloc[0]
            prev = hsgt.iloc[-1] if len(hsgt) > 1 else latest
            vol = self._safe_float(latest.get("vol"))
            ratio = self._safe_float(latest.get("ratio"))
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

    def _fetch_and_compute_rs(self, kline_bundle: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
        """获取沪深300日线并计算个股相对强弱。"""
        if not kline_bundle:
            return {"ok": False}
        day_item = kline_bundle.get("day", {})
        day_df = day_item.get("df") if isinstance(day_item, dict) else None
        if day_df is None or day_df.empty:
            return {"ok": False}
        try:
            import akshare as ak
            idx = ak.stock_zh_index_daily(symbol="sh000300")
            if idx is None or len(idx) < 60:
                return {"ok": False}
            stock_close = day_df["close"].astype(float)
            index_close = idx["close"].astype(float)
            return self._compute_relative_strength(stock_close, index_close)
        except Exception:
            return {"ok": False}

    def _fetch_industry_index_daily(self, industry_name: str):
        """获取东财行业板块指数日线收盘价序列。"""
        if not industry_name:
            return None
        try:
            import akshare as ak
            time.sleep(0.5)
            end_d = datetime.now().strftime("%Y%m%d")
            start_d = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
            df = ak.stock_board_industry_hist_em(
                symbol=industry_name, period="日k",
                start_date=start_d, end_date=end_d, adjust="",
            )
            if df is None or len(df) < 20:
                return None
            return df["收盘"].astype(float).reset_index(drop=True)
        except Exception:
            return None

    def _fetch_and_compute_rs_industry(self, kline_bundle, fundamentals):
        """计算个股 vs 行业板块指数的相对强弱。"""
        if not kline_bundle:
            return {"ok": False}
        day_df = (kline_bundle.get("day") or {}).get("df")
        if day_df is None or day_df.empty:
            return {"ok": False}
        industry = fundamentals.get("industry", "")
        if not industry:
            return {"ok": False}
        ind_close = self._fetch_industry_index_daily(industry)
        if ind_close is None:
            return {"ok": False}
        result = self._compute_relative_strength(day_df["close"].astype(float), ind_close)
        result["benchmark"] = f"行业:{industry}"
        return result

    def _fetch_sector_leaders(self, industry_name, exclude_symbol="", top_n=3):
        """获取行业板块内成交额最大的TOP N个股代码。"""
        if not industry_name:
            return []
        try:
            import akshare as ak
            time.sleep(0.5)
            df = ak.stock_board_industry_cons_em(symbol=industry_name)
            if df is None or df.empty:
                return []
            # 排除自身
            if exclude_symbol:
                df = df[df["代码"].astype(str) != exclude_symbol]
            # 按成交额降序（市值列名不稳定，成交额更可靠）
            if "成交额" in df.columns:
                df = df.sort_values("成交额", ascending=False)
            return df["代码"].astype(str).head(top_n).tolist()
        except Exception:
            return []

    def _fetch_and_compute_rs_leaders(self, kline_bundle, fundamentals, symbol=""):
        """计算个股 vs 板块龙头TOP3均值的相对强弱。"""
        if not kline_bundle:
            return {"ok": False}
        day_df = (kline_bundle.get("day") or {}).get("df")
        if day_df is None or day_df.empty:
            return {"ok": False}
        industry = fundamentals.get("industry", "")
        if not industry:
            return {"ok": False}
        codes = self._fetch_sector_leaders(industry, exclude_symbol=symbol, top_n=3)
        if not codes:
            return {"ok": False}
        try:
            import akshare as ak
            import pandas as pd
            end_d = datetime.now().strftime("%Y%m%d")
            start_d = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
            normed = []
            used_codes = []
            for code in codes:
                try:
                    time.sleep(0.5)
                    hist = ak.stock_zh_a_hist(
                        symbol=code, period="daily",
                        start_date=start_d, end_date=end_d, adjust="qfq",
                    )
                    if hist is not None and len(hist) >= 20:
                        c = hist["收盘"].astype(float).reset_index(drop=True)
                        normed.append(c / c.iloc[0])  # 归一化
                        used_codes.append(code)
                except Exception:
                    continue
            if not normed:
                return {"ok": False}
            min_len = min(len(s) for s in normed)
            aligned = [s.tail(min_len).reset_index(drop=True) for s in normed]
            avg_leader = pd.concat(aligned, axis=1).mean(axis=1)
            stock_close = day_df["close"].astype(float)
            result = self._compute_relative_strength(stock_close, avg_leader)
            result["benchmark"] = f"龙头TOP{len(used_codes)}"
            result["leader_codes"] = used_codes
            return result
        except Exception:
            return {"ok": False}

    @staticmethod
    def _match_industry_etf(industry_name):
        """模糊匹配行业名到ETF代码。"""
        if not industry_name:
            return None
        if industry_name in _INDUSTRY_ETF_MAP:
            return _INDUSTRY_ETF_MAP[industry_name]
        for key, code in _INDUSTRY_ETF_MAP.items():
            if key in industry_name or industry_name in key:
                return code
        return None

    def _fetch_and_compute_rs_etf(self, kline_bundle, fundamentals):
        """计算个股 vs 行业ETF的相对强弱。"""
        if not kline_bundle:
            return {"ok": False}
        day_df = (kline_bundle.get("day") or {}).get("df")
        if day_df is None or day_df.empty:
            return {"ok": False}
        industry = fundamentals.get("industry", "")
        etf_code = self._match_industry_etf(industry)
        if not etf_code:
            return {"ok": False, "reason": f"no_etf_for_{industry}"}
        try:
            import akshare as ak
            time.sleep(0.5)
            end_d = datetime.now().strftime("%Y%m%d")
            start_d = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
            df = ak.fund_etf_hist_em(
                symbol=etf_code, period="daily",
                start_date=start_d, end_date=end_d, adjust="",
            )
            if df is None or len(df) < 20:
                return {"ok": False}
            etf_close = df["收盘"].astype(float).reset_index(drop=True)
            stock_close = day_df["close"].astype(float)
            result = self._compute_relative_strength(stock_close, etf_close)
            result["benchmark"] = f"ETF:{etf_code}"
            result["etf_code"] = etf_code
            return result
        except Exception:
            return {"ok": False}

    @staticmethod
    def _fetch_chip_distribution(hist_df, current_price: float) -> dict[str, Any]:
        """基于日线量价数据估算筹码分布。

        原理：以成交量为权重，按价格区间统计筹码堆积。近期成交量权重更高（衰减因子）。
        """
        import numpy as np

        if hist_df is None or len(hist_df) < 60:
            return {}
        try:
            current_price = float(current_price)
            if current_price <= 0:
                return {}
        except (ValueError, TypeError):
            return {}

        df = hist_df.tail(120).copy()
        price_min = current_price * 0.7
        price_max = current_price * 1.3
        bins = np.linspace(price_min, price_max, 21)

        chip_dist = np.zeros(20)
        close_col = "close" if "close" in df.columns else "收盘"
        high_col = "high" if "high" in df.columns else "最高"
        low_col = "low" if "low" in df.columns else "最低"
        vol_col = "volume" if "volume" in df.columns else "成交量"

        for i, (_, row) in enumerate(df.iterrows()):
            decay = 0.95 ** (len(df) - 1 - i)
            try:
                avg_price = (float(row[high_col]) + float(row[low_col]) + float(row[close_col])) / 3
                vol = float(row[vol_col]) * decay
            except (ValueError, TypeError, KeyError):
                continue
            bin_idx = int(np.searchsorted(bins, avg_price)) - 1
            if 0 <= bin_idx < 20:
                chip_dist[bin_idx] += vol

        total_chips = chip_dist.sum()
        if total_chips == 0:
            return {}

        chip_pct = chip_dist / total_chips * 100
        current_bin = int(np.searchsorted(bins, current_price)) - 1
        profit_ratio = float(chip_pct[:max(0, current_bin + 1)].sum())
        top3_bins = np.argsort(chip_pct)[-3:]
        concentration = float(chip_pct[top3_bins].sum())
        bin_centers = (bins[:-1] + bins[1:]) / 2
        avg_cost = float(np.average(bin_centers, weights=chip_dist))
        trapped_ratio = 100.0 - profit_ratio
        health = min(100.0, profit_ratio * 0.4 + concentration * 0.3 + (100.0 - trapped_ratio) * 0.3)

        return {
            "profit_ratio": round(profit_ratio, 1),
            "trapped_ratio": round(trapped_ratio, 1),
            "concentration": round(concentration, 1),
            "avg_cost": round(avg_cost, 2),
            "health_score": round(health, 1),
            "current_vs_cost": round((current_price / avg_cost - 1) * 100, 1) if avg_cost > 0 else 0,
        }

    @staticmethod
    def _detect_market_regime() -> dict[str, Any]:
        """基于沪深300判断当前市场状态（bull/bear/range）。"""
        try:
            import akshare as ak
            idx = ak.stock_zh_index_daily(symbol="sh000300")
            if idx is None or len(idx) < 60:
                return {"regime": "unknown"}
            close = idx["close"].astype(float).tail(60)
            ma20 = float(close.rolling(20).mean().iloc[-1])
            ma60 = float(close.rolling(60).mean().iloc[-1])
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
                "index_ma20": round(ma20, 2),
                "index_ma60": round(ma60, 2),
                "index_ret_20d": round(ret_20d, 2),
                "index_vol_20d": round(vol_20d, 2),
            }
        except Exception:
            return {"regime": "unknown"}

    @staticmethod
    def _compute_relative_strength(stock_close_series, index_close_series) -> dict[str, Any]:
        """计算个股相对大盘指数的相对强弱指标。

        stock_close_series / index_close_series: pandas Series，按时间升序，至少20根。
        """
        import numpy as np
        result: dict[str, Any] = {"ok": False}
        try:
            stk = stock_close_series.astype(float).reset_index(drop=True)
            idx = index_close_series.astype(float).reset_index(drop=True)
            # 对齐长度（取较短的末尾对齐）
            min_len = min(len(stk), len(idx))
            if min_len < 20:
                return result
            stk = stk.tail(min_len).reset_index(drop=True)
            idx = idx.tail(min_len).reset_index(drop=True)

            # RS比率序列 = 个股 / 指数（归一化到起点=1）
            rs = (stk / stk.iloc[0]) / (idx / idx.iloc[0])

            # RS动量（5日/10日/20日）
            def _rs_mom(n: int) -> float | None:
                if len(rs) < n + 1:
                    return None
                return round(float((rs.iloc[-1] / rs.iloc[-1 - n] - 1) * 100), 2)

            # 超额收益 = 个股涨幅 - 指数涨幅
            def _excess(n: int) -> float | None:
                if len(stk) < n + 1 or len(idx) < n + 1:
                    return None
                stk_ret = (float(stk.iloc[-1]) / float(stk.iloc[-1 - n]) - 1) * 100
                idx_ret = (float(idx.iloc[-1]) / float(idx.iloc[-1 - n]) - 1) * 100
                return round(stk_ret - idx_ret, 2)

            rs_current = round(float(rs.iloc[-1]), 4)

            # RS是否创20日新高/新低
            rs_20 = rs.tail(20)
            rs_new_high = bool(rs.iloc[-1] >= rs_20.max())
            rs_new_low = bool(rs.iloc[-1] <= rs_20.min())

            # RS趋势：RS的5日均线 vs 20日均线
            rs_ma5 = float(rs.tail(5).mean())
            rs_ma20 = float(rs.tail(20).mean())
            if rs_ma5 > rs_ma20 * 1.005:
                rs_trend = "up"
            elif rs_ma5 < rs_ma20 * 0.995:
                rs_trend = "down"
            else:
                rs_trend = "flat"

            # 股价新高但RS没新高 → 看跌背离
            stk_20 = stk.tail(20)
            price_new_high = bool(stk.iloc[-1] >= stk_20.max())
            price_new_low = bool(stk.iloc[-1] <= stk_20.min())
            rs_divergence_bearish = price_new_high and not rs_new_high
            rs_divergence_bullish = rs_new_high and not price_new_high

            result = {
                "ok": True,
                "rs_current": rs_current,
                "rs_momentum_5d": _rs_mom(5),
                "rs_momentum_10d": _rs_mom(10),
                "rs_momentum_20d": _rs_mom(20),
                "excess_return_5d": _excess(5),
                "excess_return_10d": _excess(10),
                "excess_return_20d": _excess(20),
                "rs_new_high": rs_new_high,
                "rs_new_low": rs_new_low,
                "rs_trend": rs_trend,
                "price_new_high": price_new_high,
                "price_new_low": price_new_low,
                "rs_divergence_bearish": rs_divergence_bearish,
                "rs_divergence_bullish": rs_divergence_bullish,
            }
        except Exception:
            pass
        return result

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
            "roe": fundamentals.get("roe"),
            "grossprofit_margin": fundamentals.get("grossprofit_margin"),
            "netprofit_margin": fundamentals.get("netprofit_margin"),
            "debt_to_assets": fundamentals.get("debt_to_assets"),
            "revenue_yoy": fundamentals.get("revenue_yoy"),
            "netprofit_yoy": fundamentals.get("netprofit_yoy"),
            "eps": fundamentals.get("eps"),
            "cfps": fundamentals.get("cfps"),
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
            # 3-5根K线组合（三连阳/三连阴等，旧简版，向后兼容）
            k_combo = self._detect_kline_combo(open_, high, low, close, 5) if n >= 5 else None
            # 高级多形态识别（25+种典型形态，带置信度及位置权重）
            kline_patterns = self._detect_advanced_kline_patterns(open_, high, low, close, n) if n >= 2 else []
            # 连续性统计
            continuity_stats = self._compute_continuity_stats(open_, high, low, close, n) if n >= 2 else {}
            # 相邻K线关系摘要
            kline_adjacency = self._compute_kline_adjacency(open_, high, low, close, n) if n >= 3 else []
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
                    ma_system[f"ma{period}"] = {"value": 0, "pct_above": 0}

            # 背离检测（MACD/RSI顶底背离）
            divergence = self._detect_divergence(close, high, low, n)
            # 缠论信号
            chanlun = self._detect_chanlun_signals(close, high, low, n)
            # 量价关系信号
            vol_series = df["volume"].astype(float) if "volume" in df.columns else df.get("amount", close * 0 + 1).astype(float)
            # 大级别图形形态（v2: 传入volume用于突破量能验证）
            chart_patterns = self._detect_chart_patterns(close, high, low, n, volume=vol_series)
            volume_price = self._detect_volume_price_signals(close, vol_series, n)
            # 支撑阻力位（v2: 内含突破事件检测）
            support_resistance = self._detect_support_resistance(close, high, low, vol_series, n)
            # 趋势线构造（v2: 真实画线+穿越检测）
            trendlines = self._construct_trendlines(close, high, low, vol_series, n) if n >= 20 else {}

            # 数值字段：数据不足时用 0 兜底，避免下游 float(None) / f"{None:.2f}" 崩溃
            def _r(val, ndigits=4, default=0):
                return round(val, ndigits) if val is not None else default

            def _ri(seq, idx, ndigits=4, default=0):
                if seq and len(seq) > idx and seq[idx] is not None:
                    return round(seq[idx], ndigits)
                return default

            out[tf] = {
                "ok": True,
                "rows": int(n),
                "timeframe_label": {"day": "日线", "week": "周线", "month": "月线"}.get(tf, tf),
                "momentum_10": round(mom, 4),
                "volatility_20": round(vol, 4),
                "amplitude_20": round(amp, 4),
                "upper_shadow_ratio": round(upper_shadow, 4),
                "lower_shadow_ratio": round(lower_shadow, 4),
                "rsi": _r(rsi_val, 2),
                "macd_dif": _ri(macd_val, 0),
                "macd_dea": _ri(macd_val, 1),
                "macd_hist": _ri(macd_val, 2),
                "kdj_k": _ri(kdj_val, 0, 2),
                "kdj_d": _ri(kdj_val, 1, 2),
                "kdj_j": _ri(kdj_val, 2, 2),
                "boll_upper": _ri(boll_val, 0),
                "boll_mid": _ri(boll_val, 1),
                "boll_lower": _ri(boll_val, 2),
                "stoch_rsi": _r(stoch_rsi_val, 2),
                "kline_combo_5": k_combo,
                "kline_patterns": kline_patterns,
                "continuity_stats": continuity_stats,
                "kline_adjacency": kline_adjacency,
                "trend_slope_pct": _r(trend_slope),
                "ma_system": ma_system,
                "close": round(curr_price, 4),
                "divergence": divergence,
                "volume_price": volume_price,
                "support_resistance": support_resistance,
                "chanlun": chanlun,
                "chart_patterns": chart_patterns,
                "trendlines": trendlines,
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
    def _detect_advanced_kline_patterns(open_, high, low, close, n: int) -> list[dict]:
        """检测最新5根K线的25+种典型形态组合，带置信度、方向标注及位置权重。

        返回格式：[{"name": str, "direction": "bullish"|"bearish"|"neutral",
                   "confidence": int, "desc": str, "position_pct": float}]
        形态越强、置信度越高。多形态同时存在时全部返回。
        """
        import numpy as np

        lookback = min(5, n)
        o = open_.tail(lookback).values.astype(float)
        h = high.tail(lookback).values.astype(float)
        lv = low.tail(lookback).values.astype(float)
        c = close.tail(lookback).values.astype(float)

        body = c - o
        abs_body = np.abs(body)
        rng = h - lv
        rng = np.where(rng == 0, 1e-8, rng)

        # 近期趋势方向（用最后5根K线的线性斜率判断）
        xs = np.arange(len(c), dtype=float)
        slope = float(np.polyfit(xs, c, 1)[0]) if len(c) >= 3 else 0.0
        uptrend = slope > 0
        downtrend = slope < 0

        patterns: list[dict] = []

        # ── 单根K线形态（最后一根）───────────────────────────────────────────
        r = rng[-1]
        ab = abs_body[-1]
        upper_wick = h[-1] - max(o[-1], c[-1])
        lower_wick = min(o[-1], c[-1]) - lv[-1]

        if r > 0:
            # 锤子线：下影>=2×实体，上影<15%波动范围，下跌趋势末端
            if lower_wick >= 2 * max(ab, r * 0.04) and upper_wick < r * 0.15 and downtrend:
                patterns.append({"name": "锤子线", "direction": "bullish", "confidence": 74,
                                 "desc": "下跌末端出现长下影锤头，买盘支撑强，看涨反转信号"})
            # 上吊线：外形同锤头但在上涨趋势末端
            elif lower_wick >= 2 * max(ab, r * 0.04) and upper_wick < r * 0.15 and uptrend:
                patterns.append({"name": "上吊线", "direction": "bearish", "confidence": 66,
                                 "desc": "上涨末端出现长下影上吊线，获利了结压力大，看跌警示"})
            # 射击之星：上影>=2×实体，下影<15%，上涨趋势末端
            if upper_wick >= 2 * max(ab, r * 0.04) and lower_wick < r * 0.15 and uptrend:
                patterns.append({"name": "射击之星", "direction": "bearish", "confidence": 74,
                                 "desc": "高位出现长上影射击之星，上方压力强，看跌反转信号"})
            # 倒锤头：上影>=2×实体，下影<15%，下跌趋势末端
            elif upper_wick >= 2 * max(ab, r * 0.04) and lower_wick < r * 0.15 and downtrend:
                patterns.append({"name": "倒锤头", "direction": "bullish", "confidence": 61,
                                 "desc": "下跌末端出现倒锤头，下方支撑开始介入，潜在反转"})
            # 十字星/纺锤线：实体<10%波动范围，多空均衡
            if ab < r * 0.1:
                patterns.append({"name": "十字星", "direction": "neutral", "confidence": 50,
                                 "desc": "多空分歧，实体极小，市场犹豫，关注后续方向选择"})
            # 大阳线：实体>60%波动范围
            if body[-1] > r * 0.6:
                patterns.append({"name": "大阳线", "direction": "bullish", "confidence": 67,
                                 "desc": "强势阳线，多头主导，收盘在高位，延续看涨"})
            # 大阴线：实体>60%波动范围
            if body[-1] < -r * 0.6:
                patterns.append({"name": "大阴线", "direction": "bearish", "confidence": 67,
                                 "desc": "强势阴线，空头主导，收盘在低位，延续看跌"})

        # ── 两根K线形态（最后两根）──────────────────────────────────────────
        if lookback >= 2:
            b1, b2 = body[-2], body[-1]
            r1, r2 = rng[-2], rng[-1]
            ab1, ab2 = abs_body[-2], abs_body[-1]

            # 看涨吞噬：前阴后阳，阳线实体完全覆盖阴线实体
            if b1 < -r1 * 0.25 and b2 > 0 and o[-1] <= c[-2] and c[-1] >= o[-2]:
                patterns.append({"name": "看涨吞噬", "direction": "bullish", "confidence": 79,
                                 "desc": "阳线完全吞噬前阴，多头强势接管，看涨反转信号强烈"})
            # 看跌吞噬：前阳后阴，阴线实体完全覆盖阳线实体
            if b1 > r1 * 0.25 and b2 < 0 and o[-1] >= c[-2] and c[-1] <= o[-2]:
                patterns.append({"name": "看跌吞噬", "direction": "bearish", "confidence": 79,
                                 "desc": "阴线完全吞噬前阳，空头强势接管，看跌反转信号强烈"})
            # 看涨孕线：大阴线后，小阳线实体在前棒实体范围内
            if b1 < -r1 * 0.35 and b2 > 0 and o[-1] > c[-2] and c[-1] < o[-2] and ab2 < ab1 * 0.55:
                patterns.append({"name": "看涨孕线", "direction": "bullish", "confidence": 63,
                                 "desc": "小阳孕于大阴腹中，空头动能衰减，蓄势看涨反转"})
            # 看跌孕线：大阳线后，小阴线实体在前棒实体范围内
            if b1 > r1 * 0.35 and b2 < 0 and o[-1] < c[-2] and c[-1] > o[-2] and ab2 < ab1 * 0.55:
                patterns.append({"name": "看跌孕线", "direction": "bearish", "confidence": 63,
                                 "desc": "小阴孕于大阳腹中，多头动能衰减，蓄势看跌反转"})
            # 穿刺形态：阴线后低开，阳线收盘超过阴线中线
            if (b1 < -r1 * 0.3 and b2 > 0 and o[-1] < lv[-2]
                    and c[-1] > (o[-2] + c[-2]) / 2):
                patterns.append({"name": "穿刺形态", "direction": "bullish", "confidence": 73,
                                 "desc": "低开后强势回收超越阴线中线，买盘积极，看涨反转"})
            # 乌云盖顶：阳线后高开，阴线收盘跌破阳线中线
            if (b1 > r1 * 0.3 and b2 < 0 and o[-1] > h[-2]
                    and c[-1] < (o[-2] + c[-2]) / 2):
                patterns.append({"name": "乌云盖顶", "direction": "bearish", "confidence": 73,
                                 "desc": "高开后深度回落跌穿阳线中线，卖盘涌现，看跌反转"})
            # 平头顶部：两根K线高点极近，上涨趋势中
            if uptrend and abs(h[-1] - h[-2]) / max(h[-2], 1e-8) < 0.003:
                patterns.append({"name": "平头顶部", "direction": "bearish", "confidence": 65,
                                 "desc": "连续两根K线高点持平，上方阻力明确，看跌反转警示"})
            # 平头底部：两根K线低点极近，下跌趋势中
            if downtrend and abs(lv[-1] - lv[-2]) / max(lv[-2], 1e-8) < 0.003:
                patterns.append({"name": "平头底部", "direction": "bullish", "confidence": 65,
                                 "desc": "连续两根K线低点持平，下方支撑坚实，看涨反转信号"})
            # 反击线(看涨)：阴线后跳空低开，阳线收盘接近前阴收盘
            if (b1 < 0 and b2 > 0 and o[-1] < lv[-2]
                    and abs(c[-1] - c[-2]) / max(abs(c[-2]), 1e-8) < 0.003):
                patterns.append({"name": "反击线(看涨)", "direction": "bullish", "confidence": 68,
                                 "desc": "跳空低开后强力反攻，收盘回到前阴收盘位，多头反击信号"})
            # 反击线(看跌)：阳线后跳空高开，阴线收盘接近前阳收盘
            if (b1 > 0 and b2 < 0 and o[-1] > h[-2]
                    and abs(c[-1] - c[-2]) / max(abs(c[-2]), 1e-8) < 0.003):
                patterns.append({"name": "反击线(看跌)", "direction": "bearish", "confidence": 68,
                                 "desc": "跳空高开后沉重回落，收盘跌至前阳收盘位，空头反击信号"})
            # 跳空高开阳线：open[-1] > high[-2] 且 body[-1]>0，上涨延续
            if o[-1] > h[-2] and b2 > 0:
                patterns.append({"name": "跳空高开阳线", "direction": "bullish", "confidence": 72,
                                 "desc": "跳空高开且收阳，强势上涨延续，多头气势充沛"})
            # 跳空低开阴线：open[-1] < low[-2] 且 body[-1]<0，下跌延续
            if o[-1] < lv[-2] and b2 < 0:
                patterns.append({"name": "跳空低开阴线", "direction": "bearish", "confidence": 72,
                                 "desc": "跳空低开且收阴，弱势下跌延续，空头压力沉重"})

        # ── 三根K线形态（最后三根）──────────────────────────────────────────
        if lookback >= 3:
            b0, b1_v, b2_v = body[-3], body[-2], body[-1]
            r0, r1_v, r2_v = rng[-3], rng[-2], rng[-1]
            ab0, ab1_v, ab2_v = abs_body[-3], abs_body[-2], abs_body[-1]

            # 早晨之星：强阴+小实体(可含十字)+强阳，阳线收盘超前阴中线
            if (b0 < -r0 * 0.4 and ab1_v < r1_v * 0.3
                    and b2_v > r2_v * 0.4 and c[-1] > (o[-3] + c[-3]) / 2):
                patterns.append({"name": "早晨之星", "direction": "bullish", "confidence": 83,
                                 "desc": "底部经典三星反转：强阴-星线-强阳，看涨反转信号极强"})
            # 黄昏之星：强阳+小实体+强阴，阴线收盘跌破前阳中线
            if (b0 > r0 * 0.4 and ab1_v < r1_v * 0.3
                    and b2_v < -r2_v * 0.4 and c[-1] < (o[-3] + c[-3]) / 2):
                patterns.append({"name": "黄昏之星", "direction": "bearish", "confidence": 83,
                                 "desc": "顶部经典三星反转：强阳-星线-强阴，看跌反转信号极强"})
            # 红三兵（三白兵）：三根阳线逐步上行，开盘在前棒实体内
            if (b0 > r0 * 0.3 and b1_v > r1_v * 0.3 and b2_v > r2_v * 0.3
                    and c[-1] > c[-2] > c[-3]
                    and o[-2] > o[-3] and o[-1] > o[-2]):
                patterns.append({"name": "红三兵", "direction": "bullish", "confidence": 84,
                                 "desc": "三阳逐级上行，多头气势强劲，上涨趋势强力确认"})
            # 三只乌鸦：三根阴线逐步下行，开盘在前棒实体内
            if (b0 < -r0 * 0.3 and b1_v < -r1_v * 0.3 and b2_v < -r2_v * 0.3
                    and c[-1] < c[-2] < c[-3]
                    and o[-2] < o[-3] and o[-1] < o[-2]):
                patterns.append({"name": "三只乌鸦", "direction": "bearish", "confidence": 84,
                                 "desc": "三阴逐级下行，空头气势强劲，下跌趋势强力确认"})
            # 两阳夹一阴：阳-阴-阳，震荡后多头接力
            if b0 > 0 and b1_v < 0 and b2_v > 0 and c[-1] > c[-3]:
                patterns.append({"name": "两阳夹一阴", "direction": "bullish", "confidence": 64,
                                 "desc": "震荡回调后多头接力，整体仍维持看涨态势"})
            # 两阴夹一阳：阴-阳-阴，反弹后空头延续
            if b0 < 0 and b1_v > 0 and b2_v < 0 and c[-1] < c[-3]:
                patterns.append({"name": "两阴夹一阳", "direction": "bearish", "confidence": 64,
                                 "desc": "反弹后空头继续主导，下跌趋势延续信号"})
            # 三内上 Three Inside Up：看涨孕线(1-2根) + 第3根阳线收盘超过第1根开盘
            if (b0 < -r0 * 0.35 and b1_v > 0 and o[-2] > c[-3] and c[-2] < o[-3]
                    and ab1_v < ab0 * 0.55
                    and b2_v > 0 and c[-1] > o[-3]):
                patterns.append({"name": "三内上", "direction": "bullish", "confidence": 76,
                                 "desc": "看涨孕线后阳线突破确认，空头动能耗尽，可靠看涨反转"})
            # 三内下 Three Inside Down：看跌孕线(1-2根) + 第3根阴线收盘低于第1根开盘
            if (b0 > r0 * 0.35 and b1_v < 0 and o[-2] < c[-3] and c[-2] > o[-3]
                    and ab1_v < ab0 * 0.55
                    and b2_v < 0 and c[-1] < o[-3]):
                patterns.append({"name": "三内下", "direction": "bearish", "confidence": 76,
                                 "desc": "看跌孕线后阴线跌破确认，多头动能耗尽，可靠看跌反转"})
            # 三外上 Three Outside Up：看涨吞噬(1-2根) + 第3根阳线继续新高
            if (b0 < -r0 * 0.25 and b1_v > 0 and o[-2] <= c[-3] and c[-2] >= o[-3]
                    and b2_v > 0 and c[-1] > c[-2]):
                patterns.append({"name": "三外上", "direction": "bullish", "confidence": 78,
                                 "desc": "看涨吞噬后继续上攻创新高，多头强势确认，看涨延续"})
            # 三外下 Three Outside Down：看跌吞噬(1-2根) + 第3根阴线继续新低
            if (b0 > r0 * 0.25 and b1_v < 0 and o[-2] >= c[-3] and c[-2] <= o[-3]
                    and b2_v < 0 and c[-1] < c[-2]):
                patterns.append({"name": "三外下", "direction": "bearish", "confidence": 78,
                                 "desc": "看跌吞噬后继续下挫创新低，空头强势确认，看跌延续"})

        # ── 五根K线形态（最后五根）──────────────────────────────────────────
        if lookback >= 5:
            # W底/双底：两个相近低点+中间高点+最后一根阳线
            low1 = float(np.min(lv[0:2]))
            low2 = float(np.min(lv[3:5]))
            mid_high = float(np.max(h[1:4]))
            if (abs(low2 - low1) / max(abs(low1), 1e-8) < 0.025
                    and mid_high > low1 * 1.025
                    and body[-1] > 0
                    and c[-1] > mid_high * 0.97):
                patterns.append({"name": "W底双底", "direction": "bullish", "confidence": 78,
                                 "desc": "双底形态颈线附近确认，看涨反转，空头陷阱化解"})
            # M顶/双顶：两个相近高点+中间低点+最后一根阴线
            high1 = float(np.max(h[0:2]))
            high2 = float(np.max(h[3:5]))
            mid_low = float(np.min(lv[1:4]))
            if (abs(high2 - high1) / max(abs(high1), 1e-8) < 0.025
                    and mid_low < high1 * 0.975
                    and body[-1] < 0
                    and c[-1] < mid_low * 1.03):
                patterns.append({"name": "M顶双顶", "direction": "bearish", "confidence": 78,
                                 "desc": "双顶形态颈线附近确认，看跌反转，多头陷阱化解"})
            # 头肩底：头部最低，两肩近似等高
            sh1_l = float(np.min(lv[0:2]))
            head_l = float(np.min(lv[1:4]))
            sh2_l = float(np.min(lv[3:5]))
            neck_h = float(np.max(h[[1, 3]]))
            if (head_l < sh1_l * 0.985 and head_l < sh2_l * 0.985
                    and abs(sh1_l - sh2_l) / max(sh1_l, 1e-8) < 0.04
                    and c[-1] > neck_h):
                patterns.append({"name": "头肩底", "direction": "bullish", "confidence": 81,
                                 "desc": "头肩底突破颈线，底部反转确认，强烈看涨信号"})
            # 头肩顶：头部最高，两肩近似等高
            sh1_h = float(np.max(h[0:2]))
            head_h = float(np.max(h[1:4]))
            sh2_h = float(np.max(h[3:5]))
            neck_l = float(np.min(lv[[1, 3]]))
            if (head_h > sh1_h * 1.015 and head_h > sh2_h * 1.015
                    and abs(sh1_h - sh2_h) / max(sh1_h, 1e-8) < 0.04
                    and c[-1] < neck_l):
                patterns.append({"name": "头肩顶", "direction": "bearish", "confidence": 81,
                                 "desc": "头肩顶跌破颈线，顶部反转确认，强烈看跌信号"})
            # 上升三法 Rising Three Methods：大阳 + 3小阴(在第1根范围内) + 大阳突破第1根高点
            if (body[0] > rng[0] * 0.5
                    and all(body[i] < 0 and abs_body[i] < abs_body[0] * 0.4
                            and lv[i] >= lv[0] and h[i] <= h[0] for i in range(1, 4))
                    and body[4] > rng[4] * 0.5 and c[4] > h[0]):
                patterns.append({"name": "上升三法", "direction": "bullish", "confidence": 80,
                                 "desc": "大阳后三小阴回调未破底，末根大阳突破前高，强势上涨延续"})
            # 下降三法 Falling Three Methods：大阴 + 3小阳(在第1根范围内) + 大阴跌破第1根低点
            if (body[0] < -rng[0] * 0.5
                    and all(body[i] > 0 and abs_body[i] < abs_body[0] * 0.4
                            and h[i] <= h[0] and lv[i] >= lv[0] for i in range(1, 4))
                    and body[4] < -rng[4] * 0.5 and c[4] < lv[0]):
                patterns.append({"name": "下降三法", "direction": "bearish", "confidence": 80,
                                 "desc": "大阴后三小阳反弹未破顶，末根大阴跌破前低，强势下跌延续"})

        # ── 位置权重调整 ──────────────────────────────────────────────────
        # 计算当前收盘在近20根K线中的相对位置 (0=极底部, 100=极顶部)
        pos_lookback = min(20, n)
        pos_close = close.tail(pos_lookback)
        pos_high_val = float(pos_close.max()) if hasattr(pos_close, 'max') else float(np.max(pos_close.values))
        pos_low_val = float(pos_close.min()) if hasattr(pos_close, 'min') else float(np.min(pos_close.values))
        pos_range = pos_high_val - pos_low_val
        current_close = float(close.iloc[-1])
        position_pct = ((current_close - pos_low_val) / pos_range * 100) if pos_range > 0 else 50.0

        for p in patterns:
            d = p.get("direction")
            conf = p["confidence"]
            if d == "bullish":
                if position_pct < 30:
                    conf += 8
                elif position_pct > 70:
                    conf -= 10
            elif d == "bearish":
                if position_pct > 70:
                    conf += 8
                elif position_pct < 30:
                    conf -= 10
            p["confidence"] = max(30, min(95, conf))
            p["position_pct"] = round(position_pct, 1)

        # 无典型形态时返回占位
        if not patterns:
            patterns.append({"name": "无明显形态", "direction": "neutral", "confidence": 50,
                             "desc": "最近3-5根K线未见典型形态组合，维持观望",
                             "position_pct": round(position_pct, 1)})
        return patterns

    @staticmethod
    def _compute_continuity_stats(open_, high, low, close, n: int) -> dict[str, Any]:
        """分析最新K线的连续性统计特征。"""
        import numpy as np

        if n < 2:
            return {"consecutive_bull": 0, "consecutive_bear": 0, "body_trend": "stable",
                    "higher_highs": 0, "lower_lows": 0, "gap_up_count": 0, "gap_down_count": 0}

        o = open_.tail(min(20, n)).values.astype(float)
        h = high.tail(min(20, n)).values.astype(float)
        lv = low.tail(min(20, n)).values.astype(float)
        c = close.tail(min(20, n)).values.astype(float)
        body = c - o

        # 连阳/连阴天数（从最新一根往回数）
        consecutive_bull = 0
        consecutive_bear = 0
        for i in range(len(body) - 1, -1, -1):
            if body[i] > 0:
                consecutive_bull += 1
            else:
                break
        if consecutive_bull == 0:
            for i in range(len(body) - 1, -1, -1):
                if body[i] < 0:
                    consecutive_bear += 1
                else:
                    break

        # body_trend：最近3根实体绝对值是递增/递减/稳定
        body_trend = "stable"
        if len(body) >= 3:
            abs3 = np.abs(body[-3:])
            if abs3[2] > abs3[1] > abs3[0]:
                body_trend = "escalating"
            elif abs3[2] < abs3[1] < abs3[0]:
                body_trend = "de-escalating"

        # higher_highs：连续高点抬升天数
        higher_highs = 0
        for i in range(len(h) - 1, 0, -1):
            if h[i] > h[i - 1]:
                higher_highs += 1
            else:
                break

        # lower_lows：连续低点下移天数
        lower_lows = 0
        for i in range(len(lv) - 1, 0, -1):
            if lv[i] < lv[i - 1]:
                lower_lows += 1
            else:
                break

        # 最近5根中跳空次数
        last5_o = open_.tail(min(5, n)).values.astype(float)
        last5_h = high.tail(min(5, n)).values.astype(float)
        last5_lv = low.tail(min(5, n)).values.astype(float)
        last5_c = close.tail(min(5, n)).values.astype(float)
        gap_up_count = 0
        gap_down_count = 0
        # Use the full series for previous bar's high/low
        full_h = high.tail(min(6, n)).values.astype(float)
        full_lv = low.tail(min(6, n)).values.astype(float)
        if len(full_h) >= 2:
            for i in range(1, len(full_h)):
                if full_lv[i] > full_h[i - 1]:  # gap up: current low > previous high
                    gap_up_count += 1
                if full_h[i] < full_lv[i - 1]:  # gap down: current high < previous low
                    gap_down_count += 1

        return {
            "consecutive_bull": consecutive_bull,
            "consecutive_bear": consecutive_bear,
            "body_trend": body_trend,
            "higher_highs": higher_highs,
            "lower_lows": lower_lows,
            "gap_up_count": gap_up_count,
            "gap_down_count": gap_down_count,
        }

    @staticmethod
    def _compute_kline_adjacency(open_, high, low, close, n: int) -> list[dict]:
        """分析最近3根K线的相邻关系，返回2个元素描述 bar[-3]→bar[-2] 和 bar[-2]→bar[-1]。"""
        import numpy as np

        if n < 3:
            return []

        o = open_.tail(3).values.astype(float)
        h = high.tail(3).values.astype(float)
        lv = low.tail(3).values.astype(float)
        c = close.tail(3).values.astype(float)
        body = c - o

        result = []
        labels = ["倒数第3→倒数第2", "倒数第2→最新"]
        for idx in range(2):
            i, j = idx, idx + 1
            # gap_pct
            gap_pct = (o[j] - c[i]) / max(abs(c[i]), 1e-8) * 100

            # high/low change
            def _change(cur, prev):
                if abs(cur - prev) / max(abs(prev), 1e-8) < 0.001:
                    return "equal"
                return "higher" if cur > prev else "lower"

            high_change = _change(h[j], h[i])
            low_change = _change(lv[j], lv[i])

            # body overlap percentage
            top_i = max(o[i], c[i])
            bot_i = min(o[i], c[i])
            top_j = max(o[j], c[j])
            bot_j = min(o[j], c[j])
            overlap_top = min(top_i, top_j)
            overlap_bot = max(bot_i, bot_j)
            overlap = max(0, overlap_top - overlap_bot)
            union = max(top_i, top_j) - min(bot_i, bot_j)
            body_overlap_pct = round(overlap / max(union, 1e-8) * 100, 1)

            # engulf degree: +1 = j fully engulfs i, -1 = j fully inside i
            body_i = abs(body[i])
            body_j = abs(body[j])
            if body_i < 1e-8 and body_j < 1e-8:
                engulf_degree = 0.0
            elif body_i < 1e-8:
                engulf_degree = 1.0
            elif body_j < 1e-8:
                engulf_degree = -1.0
            else:
                engulf_degree = round((body_j - body_i) / max(body_i, body_j), 2)

            # human-readable relationship
            dir_i = "阳" if body[i] > 0 else ("阴" if body[i] < 0 else "十字")
            dir_j = "阳" if body[j] > 0 else ("阴" if body[j] < 0 else "十字")
            parts = [f"{dir_i}→{dir_j}"]
            if abs(gap_pct) > 0.1:
                parts.append(f"跳空{gap_pct:+.1f}%{'高' if gap_pct > 0 else '低'}开")
            if high_change == "higher":
                parts.append("高点抬升")
            elif high_change == "lower":
                parts.append("高点回落")
            if engulf_degree > 0.3:
                parts.append(f"吞噬度{engulf_degree:.1f}")
            elif engulf_degree < -0.3:
                parts.append("被包含")
            relationship = ", ".join(parts)

            result.append({
                "pair": labels[idx],
                "gap_pct": round(gap_pct, 2),
                "high_change": high_change,
                "low_change": low_change,
                "body_overlap_pct": body_overlap_pct,
                "engulf_degree": engulf_degree,
                "relationship": relationship,
            })
        return result

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

    # ── 突破检测公共模块 (v2) ──

    @staticmethod
    def _confirm_breakout(close, high, low, volume, level: float,
                          direction: str = "up", confirm_bars: int = 2,
                          vr_threshold: float = 1.3) -> dict[str, Any]:
        """判断价格是否有效突破某个关键价位。

        Args:
            close/high/low/volume: K线数据数组
            level: 关键价位（颈线/趋势线/支撑阻力）
            direction: "up"=向上突破阻力, "down"=向下跌破支撑
            confirm_bars: 需要连续站稳的K线根数
            vr_threshold: 量能确认阈值（突破根量/前20均量）
        Returns:
            dict with confirmed, strength, bars_above, volume_ratio, type
        """
        import numpy as np

        n = len(close)
        result = {
            "confirmed": False, "strength": 0, "bars_above": 0,
            "volume_ratio": 0.0, "type": "none",
            "level": round(float(level), 2),
        }
        if n < 5:
            return result

        c = close[-20:] if len(close) > 20 else close
        h = high[-20:] if len(high) > 20 else high
        l = low[-20:] if len(low) > 20 else low
        v = volume[-20:] if len(volume) > 20 else volume

        curr = float(c[-1])
        avg_vol = float(np.mean(v[:-1])) if len(v) > 1 else 1.0

        if direction == "up":
            # 计算连续站稳根数
            bars_above = 0
            for i in range(len(c) - 1, -1, -1):
                if float(c[i]) > level:
                    bars_above += 1
                else:
                    break

            # 突破根的量比
            break_idx = len(c) - bars_above  # 突破那根K线
            if break_idx < len(v) and avg_vol > 0:
                result["volume_ratio"] = round(float(v[break_idx]) / avg_vol, 2)

            result["bars_above"] = bars_above

            if bars_above == 0:
                # 检查是否影线试探
                if float(h[-1]) > level > curr:
                    result["type"] = "wick_only"
                    result["strength"] = 10
                return result

            # 实体确认：突破K线的实体（非影线）是否穿过价位
            body_cross = False
            if break_idx < len(c):
                open_price = float(c[break_idx - 1]) if break_idx > 0 else float(c[break_idx])
                body_low = min(open_price, float(c[break_idx]))
                if body_low < level < float(c[break_idx]):
                    body_cross = True

            # 回踩验证
            has_retest = False
            if bars_above >= 3:
                for i in range(len(c) - bars_above, len(c)):
                    if float(l[i]) <= level * 1.01 and float(c[i]) > level:
                        has_retest = True
                        break

            # 综合评分
            strength = 0
            strength += min(30, bars_above * 10)  # 站稳根数 max 30
            strength += 20 if body_cross else 0    # 实体穿越 20
            strength += 20 if result["volume_ratio"] >= vr_threshold else 0  # 量能 20
            strength += 15 if has_retest else 0    # 回踩确认 15
            strength += 15 if curr > level * 1.02 else 0  # 距离2%以上 15

            result["strength"] = int(min(100, strength))
            result["confirmed"] = bool(bars_above >= confirm_bars)
            if has_retest:
                result["type"] = "with_retest"
            elif result["confirmed"]:
                result["type"] = "clean"
            else:
                result["type"] = "fresh"

        elif direction == "down":
            # 向下跌破，对称逻辑
            bars_below = 0
            for i in range(len(c) - 1, -1, -1):
                if float(c[i]) < level:
                    bars_below += 1
                else:
                    break

            break_idx = len(c) - bars_below
            if break_idx < len(v) and avg_vol > 0:
                result["volume_ratio"] = round(float(v[break_idx]) / avg_vol, 2)

            result["bars_above"] = bars_below  # 复用字段名，含义为bars_below

            if bars_below == 0:
                if float(l[-1]) < level < curr:
                    result["type"] = "wick_only"
                    result["strength"] = 10
                return result

            body_cross = False
            if break_idx < len(c):
                open_price = float(c[break_idx - 1]) if break_idx > 0 else float(c[break_idx])
                body_high = max(open_price, float(c[break_idx]))
                if float(c[break_idx]) < level < body_high:
                    body_cross = True

            has_retest = False
            if bars_below >= 3:
                for i in range(len(c) - bars_below, len(c)):
                    if float(h[i]) >= level * 0.99 and float(c[i]) < level:
                        has_retest = True
                        break

            strength = 0
            strength += min(30, bars_below * 10)
            strength += 20 if body_cross else 0
            strength += 20 if result["volume_ratio"] >= vr_threshold else 0
            strength += 15 if has_retest else 0
            strength += 15 if curr < level * 0.98 else 0

            result["strength"] = int(min(100, strength))
            result["confirmed"] = bool(bars_below >= confirm_bars)
            if has_retest:
                result["type"] = "with_retest"
            elif result["confirmed"]:
                result["type"] = "clean"
            else:
                result["type"] = "fresh"

        return result

    @staticmethod
    def _construct_trendlines(close, high, low, volume, n: int) -> dict[str, Any]:
        """构造真实趋势线：用局部极值点连线，检测价格穿越。

        上升趋势线：连接波谷（局部低点），价格跌破=利空
        下降趋势线：连接波峰（局部高点），价格突破=利多
        """
        import numpy as np

        result: dict[str, Any] = {
            "up_trendline": None,
            "down_trendline": None,
        }
        lookback = min(60, n)
        if lookback < 20:
            return result

        c = close.values[-lookback:].astype(float)
        h = high.values[-lookback:].astype(float)
        l = low.values[-lookback:].astype(float)
        v = volume.values[-lookback:].astype(float)

        # 找局部极值点（窗口=5）
        win = 5
        peaks, troughs = [], []
        for i in range(win, lookback - win):
            if h[i] == max(h[i-win:i+win+1]):
                peaks.append(i)
            if l[i] == min(l[i-win:i+win+1]):
                troughs.append(i)

        # --- 上升趋势线（连接波谷）---
        if len(troughs) >= 2:
            best_line = None
            best_touches = 0
            # 尝试不同的两点组合，找触碰最多的线
            for i in range(len(troughs)):
                for j in range(i + 1, len(troughs)):
                    x1, y1 = troughs[i], l[troughs[i]]
                    x2, y2 = troughs[j], l[troughs[j]]
                    if y2 <= y1:
                        continue  # 上升趋势线斜率必须>0
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - slope * x1
                    # 计算触碰点（低点在线附近1%以内）
                    touches = 0
                    all_above = True
                    for t in troughs:
                        line_val = slope * t + intercept
                        diff_pct = (l[t] - line_val) / line_val * 100 if line_val else 0
                        if -1.0 <= diff_pct <= 1.5:
                            touches += 1
                        if l[t] < line_val * 0.97:
                            all_above = False
                    if touches >= 2 and all_above and touches > best_touches:
                        best_touches = touches
                        best_line = (slope, intercept, touches)

            if best_line:
                slope, intercept, touches = best_line
                line_now = slope * (lookback - 1) + intercept
                curr_price = c[-1]
                pct_vs_line = (curr_price - line_now) / line_now * 100 if line_now else 0
                broken = bool(curr_price < line_now * 0.99)  # 跌破1%以上算破位

                # 如果跌破，用_confirm_breakout验证
                breakout_info = None
                if broken:
                    breakout_info = DataBackend._confirm_breakout(
                        c, h, l, v, line_now, direction="down")

                result["up_trendline"] = {
                    "slope_per_bar": round(slope, 4),
                    "current_line_price": round(line_now, 2),
                    "touch_points": touches,
                    "price_vs_line_pct": round(pct_vs_line, 2),
                    "broken": broken,
                    "breakout": breakout_info,
                }

        # --- 下降趋势线（连接波峰）---
        if len(peaks) >= 2:
            best_line = None
            best_touches = 0
            for i in range(len(peaks)):
                for j in range(i + 1, len(peaks)):
                    x1, y1 = peaks[i], h[peaks[i]]
                    x2, y2 = peaks[j], h[peaks[j]]
                    if y2 >= y1:
                        continue  # 下降趋势线斜率必须<0
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - slope * x1
                    touches = 0
                    all_below = True
                    for p in peaks:
                        line_val = slope * p + intercept
                        diff_pct = (h[p] - line_val) / line_val * 100 if line_val else 0
                        if -1.5 <= diff_pct <= 1.0:
                            touches += 1
                        if h[p] > line_val * 1.03:
                            all_below = False
                    if touches >= 2 and all_below and touches > best_touches:
                        best_touches = touches
                        best_line = (slope, intercept, touches)

            if best_line:
                slope, intercept, touches = best_line
                line_now = slope * (lookback - 1) + intercept
                curr_price = c[-1]
                pct_vs_line = (curr_price - line_now) / line_now * 100 if line_now else 0
                broken = bool(curr_price > line_now * 1.01)

                breakout_info = None
                if broken:
                    breakout_info = DataBackend._confirm_breakout(
                        c, h, l, v, line_now, direction="up")

                result["down_trendline"] = {
                    "slope_per_bar": round(slope, 4),
                    "current_line_price": round(line_now, 2),
                    "touch_points": touches,
                    "price_vs_line_pct": round(pct_vs_line, 2),
                    "broken": broken,
                    "breakout": breakout_info,
                }

        return result

    @staticmethod
    def _detect_divergence(close, high, low, n: int) -> dict[str, Any]:
        """检测 MACD 和 RSI 的顶底背离信号。

        通过比较最近两个价格极值点与对应指标值的方向关系来判断背离：
        - 顶背离：价格创新高但指标未创新高 → 看跌信号
        - 底背离：价格创新低但指标未创新低 → 看涨信号

        返回 dict 包含 macd_divergence, rsi_divergence, summary 等字段。
        """
        import numpy as np

        result: dict[str, Any] = {
            "macd_top_div": False, "macd_bot_div": False,
            "rsi_top_div": False, "rsi_bot_div": False,
            "macd_div_desc": "", "rsi_div_desc": "",
            "divergence_score": 0,  # -100~+100, 正=看涨背离, 负=看跌背离
        }
        if n < 40:
            return result

        close_arr = close.values.astype(float)
        high_arr = high.values.astype(float)
        low_arr = low.values.astype(float)

        # --- 计算 MACD DIF 序列 ---
        ema12 = close.ewm(span=12, adjust=False).mean().values
        ema26 = close.ewm(span=26, adjust=False).mean().values
        dif = ema12 - ema26

        # --- 计算 RSI(14) 序列 ---
        delta = np.diff(close_arr, prepend=close_arr[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        # EMA 方式计算 RSI
        avg_gain = np.zeros_like(gain)
        avg_loss = np.zeros_like(loss)
        avg_gain[14] = gain[1:15].mean()
        avg_loss[14] = loss[1:15].mean()
        for i in range(15, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
        rsi = 100 - 100 / (1 + rs)

        # --- 查找局部极值点（前后 window 根内的最高/最低）---
        window = max(5, n // 20)  # 自适应窗口
        lookback = min(n, 120)  # 最多回看120根
        start = max(0, n - lookback)

        def find_peaks(arr, start_idx, win):
            peaks = []
            for i in range(start_idx + win, len(arr) - 1):
                seg = arr[max(start_idx, i - win): min(len(arr), i + win + 1)]
                if len(seg) > 0 and arr[i] == seg.max() and arr[i] > arr[i-1] and arr[i] >= arr[min(i+1, len(arr)-1)]:
                    peaks.append(i)
            return peaks

        def find_troughs(arr, start_idx, win):
            troughs = []
            for i in range(start_idx + win, len(arr) - 1):
                seg = arr[max(start_idx, i - win): min(len(arr), i + win + 1)]
                if len(seg) > 0 and arr[i] == seg.min() and arr[i] < arr[i-1] and arr[i] <= arr[min(i+1, len(arr)-1)]:
                    troughs.append(i)
            return troughs

        price_peaks = find_peaks(high_arr, start, window)
        price_troughs = find_troughs(low_arr, start, window)

        div_score = 0

        # --- MACD 背离检测 ---
        if len(price_peaks) >= 2:
            p1, p2 = price_peaks[-2], price_peaks[-1]
            if high_arr[p2] > high_arr[p1] and dif[p2] < dif[p1]:
                result["macd_top_div"] = True
                result["macd_div_desc"] = f"MACD顶背离: 价格新高{high_arr[p2]:.2f}>{high_arr[p1]:.2f}, DIF走低{dif[p2]:.4f}<{dif[p1]:.4f}"
                div_score -= 30

        if len(price_troughs) >= 2:
            t1, t2 = price_troughs[-2], price_troughs[-1]
            if low_arr[t2] < low_arr[t1] and dif[t2] > dif[t1]:
                result["macd_bot_div"] = True
                result["macd_div_desc"] = f"MACD底背离: 价格新低{low_arr[t2]:.2f}<{low_arr[t1]:.2f}, DIF走高{dif[t2]:.4f}>{dif[t1]:.4f}"
                div_score += 30

        # --- RSI 背离检测 ---
        if len(price_peaks) >= 2:
            p1, p2 = price_peaks[-2], price_peaks[-1]
            if high_arr[p2] > high_arr[p1] and rsi[p2] < rsi[p1] - 2:
                result["rsi_top_div"] = True
                result["rsi_div_desc"] = f"RSI顶背离: 价格新高, RSI走低{rsi[p2]:.1f}<{rsi[p1]:.1f}"
                div_score -= 25

        if len(price_troughs) >= 2:
            t1, t2 = price_troughs[-2], price_troughs[-1]
            if low_arr[t2] < low_arr[t1] and rsi[t2] > rsi[t1] + 2:
                result["rsi_bot_div"] = True
                result["rsi_div_desc"] = f"RSI底背离: 价格新低, RSI走高{rsi[t2]:.1f}>{rsi[t1]:.1f}"
                div_score += 25

        # 双重背离加强信号
        if result["macd_top_div"] and result["rsi_top_div"]:
            div_score -= 15  # 额外扣分
        if result["macd_bot_div"] and result["rsi_bot_div"]:
            div_score += 15  # 额外加分

        result["divergence_score"] = max(-100, min(100, div_score))
        return result

    @staticmethod
    def _detect_volume_price_signals(close, volume, n: int) -> dict[str, Any]:
        """检测量价关系信号：放量突破、缩量回踩、量能异动、天量滞涨、OBV趋势。

        返回 dict 包含各量价信号及综合评分。
        """
        import numpy as np

        result: dict[str, Any] = {
            "volume_breakout": False,       # 放量突破
            "shrink_pullback": False,        # 缩量回踩
            "volume_anomaly": False,         # 底部放量异动
            "climax_volume": False,          # 天量滞涨/见顶
            "obv_trend": "neutral",          # OBV趋势方向
            "volume_price_score": 0,         # -50~+50
            "desc": "",
        }
        if n < 25:
            return result

        c = close.values.astype(float)
        v = volume.values.astype(float)
        vp_score = 0
        descs = []

        # 量比：近5日均量 / 近20日均量
        vol_5 = v[-5:].mean() if len(v) >= 5 else v.mean()
        vol_20 = v[-20:].mean() if len(v) >= 20 else v.mean()
        vr = vol_5 / vol_20 if vol_20 > 0 else 1.0

        # 最新一根的成交量 / 20日均量
        last_vr = v[-1] / vol_20 if vol_20 > 0 else 1.0

        # --- 放量突破：价格突破20日高点 且 量比>1.5 ---
        high_20 = c[-21:-1].max() if len(c) >= 21 else c[:-1].max()
        if c[-1] > high_20 and last_vr > 1.5:
            result["volume_breakout"] = True
            vp_score += 20
            descs.append(f"放量突破: 价格{c[-1]:.2f}突破20日高点{high_20:.2f}, 量比{last_vr:.1f}倍")

        # --- 缩量回踩：近3日价格回调但成交量持续萎缩 ---
        if len(c) >= 6:
            price_down = c[-1] < c[-4]  # 近3日回调
            vol_shrink = all(v[-i] < v[-i-1] for i in range(1, min(4, len(v))))  # 逐日缩量
            if price_down and vol_shrink and vr < 0.8:
                result["shrink_pullback"] = True
                vp_score += 12
                descs.append("缩量回踩: 价格回调但量能逐日萎缩,洗盘特征")

        # --- 底部放量异动：60日最低附近突然放量 ---
        if len(c) >= 60:
            low_60 = c[-60:].min()
            near_bottom = (c[-1] - low_60) / (c[-60:].max() - low_60 + 1e-8) < 0.25
            if near_bottom and last_vr > 2.0:
                result["volume_anomaly"] = True
                vp_score += 18
                descs.append(f"底部放量异动: 价格位于60日低位区间, 量比{last_vr:.1f}倍异动")

        # --- 天量滞涨/见顶：成交量创20日新高但涨幅微弱或下跌 ---
        if len(v) >= 20:
            is_max_vol = v[-1] >= v[-20:].max() * 0.95
            weak_price = abs(c[-1] / c[-2] - 1) < 0.01 or c[-1] < c[-2]
            near_top = (c[-1] - c[-60:].min()) / (c[-60:].max() - c[-60:].min() + 1e-8) > 0.8 if len(c) >= 60 else False
            if is_max_vol and weak_price and near_top:
                result["climax_volume"] = True
                vp_score -= 20
                descs.append("天量滞涨: 成交量创新高但价格停滞于高位,出货嫌疑")

        # --- OBV 能量潮趋势 ---
        if len(c) >= 30:
            pct = np.diff(c, prepend=c[0])
            obv = np.cumsum(np.where(pct > 0, v, np.where(pct < 0, -v, 0)))
            obv_20 = obv[-20:]
            obv_slope = np.polyfit(np.arange(len(obv_20)), obv_20, 1)[0]
            price_slope = np.polyfit(np.arange(20), c[-20:], 1)[0]
            if obv_slope > 0 and price_slope > 0:
                result["obv_trend"] = "bullish"
                vp_score += 8
            elif obv_slope < 0 and price_slope < 0:
                result["obv_trend"] = "bearish"
                vp_score -= 8
            elif obv_slope > 0 and price_slope < 0:
                result["obv_trend"] = "bullish_divergence"
                vp_score += 12
                descs.append("OBV底背离: 价格下跌但OBV上行,资金暗中吸筹")
            elif obv_slope < 0 and price_slope > 0:
                result["obv_trend"] = "bearish_divergence"
                vp_score -= 12
                descs.append("OBV顶背离: 价格上涨但OBV下行,资金暗中出货")

        result["volume_price_score"] = max(-50, min(50, vp_score))
        result["desc"] = "; ".join(descs) if descs else "量价关系正常"
        return result

    @staticmethod
    def _detect_support_resistance(close, high, low, volume, n: int) -> dict[str, Any]:
        """检测关键支撑位和阻力位。

        方法：前高/前低、密集成交区(量价分布)、均线支撑/压力、跳空缺口、整数关口。
        返回 sorted 支撑位列表和阻力位列表及综合评估。
        """
        import numpy as np

        result: dict[str, Any] = {
            "support_levels": [],
            "resistance_levels": [],
            "nearest_support": None,
            "nearest_resistance": None,
            "price_position": "middle",  # "near_support" | "near_resistance" | "middle"
            "sr_score": 0,  # -30~+30
            "gaps": [],
            "desc": "",
        }
        if n < 30:
            return result

        c = close.values.astype(float)
        h = high.values.astype(float)
        l = low.values.astype(float)
        v = volume.values.astype(float)
        curr = c[-1]
        supports = []
        resistances = []
        descs = []

        # --- 1. 前高/前低(波段极值) ---
        window = max(5, n // 15)
        lookback = min(n, 120)
        start = max(0, n - lookback)
        for i in range(start + window, n - 1):
            seg_h = h[max(start, i-window): min(n, i+window+1)]
            seg_l = l[max(start, i-window): min(n, i+window+1)]
            if h[i] == seg_h.max() and h[i] > h[i-1]:
                level = float(h[i])
                if level > curr:
                    resistances.append(("前高", level))
                else:
                    supports.append(("前高回落", level))
            if l[i] == seg_l.min() and l[i] < l[i-1]:
                level = float(l[i])
                if level < curr:
                    supports.append(("前低", level))
                else:
                    resistances.append(("前低反弹", level))

        # --- 2. 均线支撑/压力 ---
        for period, label in [(20, "MA20"), (60, "MA60"), (120, "MA120"), (250, "MA250")]:
            if n >= period:
                ma_val = float(c[-period:].mean())
                if ma_val < curr and (curr - ma_val) / curr < 0.05:
                    supports.append((label, round(ma_val, 2)))
                elif ma_val > curr and (ma_val - curr) / curr < 0.05:
                    resistances.append((label, round(ma_val, 2)))

        # --- 3. 密集成交区 (Volume Profile 简化版) ---
        if n >= 60:
            price_range = h[-60:].max() - l[-60:].min()
            if price_range > 0:
                num_bins = 20
                bin_edges = np.linspace(l[-60:].min(), h[-60:].max(), num_bins + 1)
                vol_profile = np.zeros(num_bins)
                for i in range(-60, 0):
                    for j in range(num_bins):
                        if l[i] <= bin_edges[j+1] and h[i] >= bin_edges[j]:
                            vol_profile[j] += v[i]
                # 找成交量最密集的价格区间
                top_bins = np.argsort(vol_profile)[-3:]
                for b in top_bins:
                    mid_price = (bin_edges[b] + bin_edges[b+1]) / 2
                    if mid_price < curr and (curr - mid_price) / curr < 0.08:
                        supports.append(("密集成交区", round(float(mid_price), 2)))
                    elif mid_price > curr and (mid_price - curr) / curr < 0.08:
                        resistances.append(("密集成交区", round(float(mid_price), 2)))

        # --- 4. 跳空缺口 ---
        gaps = []
        for i in range(max(1, n-60), n):
            if l[i] > h[i-1]:  # 向上跳空
                gaps.append({"type": "up", "top": float(l[i]), "bottom": float(h[i-1])})
                if h[i-1] < curr:
                    supports.append(("上跳缺口上沿", round(float(h[i-1]), 2)))
            elif h[i] < l[i-1]:  # 向下跳空
                gaps.append({"type": "down", "top": float(l[i-1]), "bottom": float(h[i])})
                if l[i-1] > curr:
                    resistances.append(("下跳缺口下沿", round(float(l[i-1]), 2)))
        result["gaps"] = gaps[-5:]  # 最近5个缺口

        # --- 5. 整数关口 ---
        magnitude = 10 ** max(0, int(np.log10(curr + 1e-8)) - 1)
        round_level = round(curr / magnitude) * magnitude
        for mul in [-1, 0, 1]:
            rl = round_level + mul * magnitude
            if rl > 0 and abs(rl - curr) / curr < 0.06:
                if rl < curr:
                    supports.append(("整数关口", rl))
                elif rl > curr:
                    resistances.append(("整数关口", rl))

        # --- 去重与排序 ---
        seen_s, seen_r = set(), set()
        unique_supports, unique_resistances = [], []
        for label, price in sorted(supports, key=lambda x: x[1], reverse=True):
            rp = round(price, 2)
            if rp not in seen_s:
                seen_s.add(rp)
                unique_supports.append({"label": label, "price": rp})
        for label, price in sorted(resistances, key=lambda x: x[1]):
            rp = round(price, 2)
            if rp not in seen_r:
                seen_r.add(rp)
                unique_resistances.append({"label": label, "price": rp})

        result["support_levels"] = unique_supports[:8]
        result["resistance_levels"] = unique_resistances[:8]

        # 最近支撑/阻力
        if unique_supports:
            result["nearest_support"] = unique_supports[0]
        if unique_resistances:
            result["nearest_resistance"] = unique_resistances[0]

        # 评估价格位置
        sr_score = 0
        if unique_supports and unique_resistances:
            ns = unique_supports[0]["price"]
            nr = unique_resistances[0]["price"]
            total_range = nr - ns if nr > ns else 1
            pos_ratio = (curr - ns) / total_range
            if pos_ratio < 0.25:
                result["price_position"] = "near_support"
                sr_score += 15
                descs.append(f"价格{curr:.2f}接近支撑位{ns:.2f}")
            elif pos_ratio > 0.75:
                result["price_position"] = "near_resistance"
                sr_score -= 15
                descs.append(f"价格{curr:.2f}接近阻力位{nr:.2f}")
        elif unique_supports and not unique_resistances:
            sr_score += 10
            descs.append("上方无明显阻力位")
        elif unique_resistances and not unique_supports:
            sr_score -= 10
            descs.append("下方无明显支撑位")

        # --- 6. 突破事件检测 (v2) ---
        breakout_events = []
        # 检查最近3根K线是否突破了任何阻力位
        for sr_item in unique_resistances[:3]:
            lvl = sr_item["price"]
            if curr > lvl:
                bo = DataBackend._confirm_breakout(c, h, l, v, lvl, "up")
                if bo["strength"] > 0:
                    breakout_events.append({
                        "level": lvl, "level_type": sr_item["label"],
                        "direction": "bullish", **bo,
                    })
                    if bo["confirmed"]:
                        sr_score += 12
                        descs.append(f"向上突破{sr_item['label']}{lvl:.2f}(强度{bo['strength']})")
                    elif bo["type"] == "fresh":
                        sr_score += 5
        # 检查是否跌破支撑位
        for sr_item in unique_supports[:3]:
            lvl = sr_item["price"]
            if curr < lvl:
                bo = DataBackend._confirm_breakout(c, h, l, v, lvl, "down")
                if bo["strength"] > 0:
                    breakout_events.append({
                        "level": lvl, "level_type": sr_item["label"],
                        "direction": "bearish", **bo,
                    })
                    if bo["confirmed"]:
                        sr_score -= 12
                        descs.append(f"向下跌破{sr_item['label']}{lvl:.2f}(强度{bo['strength']})")
                    elif bo["type"] == "fresh":
                        sr_score -= 5

        result["breakout_events"] = breakout_events
        result["sr_score"] = max(-30, min(30, sr_score))
        result["desc"] = "; ".join(descs) if descs else f"当前价{curr:.2f}处于支撑阻力区间中部"
        return result

    @staticmethod
    def _detect_chart_patterns(close, high, low, n: int, volume=None) -> dict[str, Any]:
        """检测大级别图形形态（20-60根K线），包括三角形、旗形、楔形、箱体、杯柄、圆弧底/顶。

        v2增强：三角形/W底/M顶/头肩的突破确认 + 量能验证。
        返回 dict 包含检测到的形态列表及综合评分。
        """
        import numpy as np

        result: dict[str, Any] = {
            "patterns": [],        # [{name, direction, confidence, desc}]
            "chart_pattern_score": 0,  # -40~+40
            "desc": "",
        }
        if n < 25:
            return result

        h = high.values.astype(float)
        l = low.values.astype(float)
        c = close.values.astype(float)
        v = volume.values.astype(float) if volume is not None else np.ones(len(c))
        patterns = []
        score = 0

        # 使用最近60根K线分析
        lookback = min(60, n)
        h60 = h[-lookback:]
        l60 = l[-lookback:]
        c60 = c[-lookback:]
        v60 = v[-lookback:]
        x = np.arange(lookback)

        # --- 1. 三角形（收敛形态）+ 突破确认 ---
        if lookback >= 30:
            # 高点趋势线（线性回归）
            high_slope = np.polyfit(x, h60, 1)[0]
            low_slope = np.polyfit(x, l60, 1)[0]
            high_line_now = np.polyval(np.polyfit(x, h60, 1), lookback - 1)
            low_line_now = np.polyval(np.polyfit(x, l60, 1), lookback - 1)

            tri_type = None
            if high_slope < 0 and low_slope > 0:
                tri_type = "symmetric"
            elif high_slope < -0.001 and abs(low_slope) < abs(high_slope) * 0.3:
                tri_type = "descending"
            elif low_slope > 0.001 and abs(high_slope) < abs(low_slope) * 0.3:
                tri_type = "ascending"

            if tri_type:
                # 判断是否已突破三角形边界
                breakout = None
                if c60[-1] > high_line_now:
                    breakout = DataBackend._confirm_breakout(c60, h60, l60, v60, high_line_now, "up")
                elif c60[-1] < low_line_now:
                    breakout = DataBackend._confirm_breakout(c60, h60, l60, v60, low_line_now, "down")

                # 收敛度（两线距离/均价）
                convergence = (high_line_now - low_line_now) / c60.mean() * 100 if c60.mean() else 0

                if tri_type == "symmetric":
                    p = {"name": "对称三角形", "direction": "neutral", "confidence": 65,
                         "desc": f"价格区间收窄至{convergence:.1f}%,即将选择方向突破"}
                    if breakout and breakout["confirmed"]:
                        p["direction"] = "bullish" if c60[-1] > high_line_now else "bearish"
                        p["confidence"] = 80
                        score += 18 if p["direction"] == "bullish" else -18
                        p["desc"] += f",已{p['direction']}突破(强度{breakout['strength']})"
                elif tri_type == "descending":
                    p = {"name": "下降三角形", "direction": "bearish", "confidence": 68,
                         "desc": "高点持续下降而低点水平支撑"}
                    score -= 12
                    if breakout and breakout["confirmed"] and c60[-1] < low_line_now:
                        score -= 10  # 向下确认突破额外减分
                        p["confidence"] = 82
                        p["desc"] += f",已向下突破(强度{breakout['strength']})"
                    elif breakout and breakout["confirmed"] and c60[-1] > high_line_now:
                        p["direction"] = "bullish"  # 反向突破
                        score += 20
                        p["confidence"] = 78
                        p["desc"] += f",反向向上突破(强度{breakout['strength']})"
                else:  # ascending
                    p = {"name": "上升三角形", "direction": "bullish", "confidence": 70,
                         "desc": "低点持续抬升逼近水平压力位"}
                    score += 15
                    if breakout and breakout["confirmed"] and c60[-1] > high_line_now:
                        score += 10  # 向上确认突破额外加分
                        p["confidence"] = 85
                        p["desc"] += f",已向上突破(强度{breakout['strength']})"

                if breakout:
                    p["breakout"] = breakout
                p["convergence_pct"] = round(convergence, 1)
                patterns.append(p)

        # --- 2. 箱体震荡 ---
        if lookback >= 30:
            range_high = h60.max()
            range_low = l60.min()
            price_range = range_high - range_low
            mean_price = c60.mean()
            # 箱体条件：振幅<15%，且最近价格在箱体中间
            if price_range / mean_price < 0.15 and abs(np.polyfit(x, c60, 1)[0]) < mean_price * 0.001:
                upper = h60[-lookback//2:].max()
                lower = l60[-lookback//2:].min()
                pos = (c60[-1] - lower) / (upper - lower) if upper > lower else 0.5
                direction = "bullish" if pos < 0.3 else ("bearish" if pos > 0.7 else "neutral")
                patterns.append({
                    "name": "箱体震荡",
                    "direction": direction,
                    "confidence": 72,
                    "desc": f"价格在[{lower:.2f}-{upper:.2f}]区间震荡,当前位于{'底部' if pos < 0.3 else '顶部' if pos > 0.7 else '中部'}",
                })
                if pos < 0.3:
                    score += 10
                elif pos > 0.7:
                    score -= 10

        # --- 3. 旗形/楔形 ---
        if lookback >= 30:
            # 先检查前半段是否有明显趋势（急涨/急跌=旗杆）
            mid = lookback // 2
            first_half_move = (c60[mid] - c60[0]) / c60[0] * 100
            second_half_slope = np.polyfit(np.arange(lookback - mid), c60[mid:], 1)[0]
            second_half_range = (h60[mid:].max() - l60[mid:].min()) / c60[mid:].mean() * 100

            if first_half_move > 15 and second_half_range < 8 and second_half_slope <= 0:
                patterns.append({
                    "name": "上升旗形",
                    "direction": "bullish",
                    "confidence": 71,
                    "desc": f"急涨{first_half_move:.1f}%后进入窄幅回调整理,旗形向上突破概率大",
                })
                score += 14
            elif first_half_move < -15 and second_half_range < 8 and second_half_slope >= 0:
                patterns.append({
                    "name": "下降旗形",
                    "direction": "bearish",
                    "confidence": 71,
                    "desc": f"急跌{abs(first_half_move):.1f}%后进入窄幅反弹整理,旗形向下突破概率大",
                })
                score -= 14

        # --- 4. 杯柄形态（Cup & Handle）---
        if lookback >= 40:
            # 找U型底：前1/3下跌，中1/3触底，后1/3回升
            third = lookback // 3
            seg1 = c60[:third]
            seg2 = c60[third:2*third]
            seg3 = c60[2*third:]
            if (seg1[0] > seg2.min() and seg3[-1] > seg2.min()
                and seg3[-1] > seg1[0] * 0.95  # 杯口基本持平
                and seg2.min() < seg1[0] * 0.9):  # 杯底深度>10%
                # 检查柄部（最近10%K线的小幅回调）
                handle = c60[-lookback//10:]
                if len(handle) >= 3 and handle[-1] < handle.max() and handle[-1] > seg2.min():
                    patterns.append({
                        "name": "杯柄形态",
                        "direction": "bullish",
                        "confidence": 75,
                        "desc": f"U型底部{seg2.min():.2f}后回升至杯口,柄部回调中,突破即买入信号",
                    })
                    score += 18

        # --- 5. 圆弧底/圆弧顶 ---
        if lookback >= 40:
            # 圆弧底：拟合二次曲线，开口向上（a>0）且近期在上升段
            coeffs = np.polyfit(x, c60, 2)
            a, b = coeffs[0], coeffs[1]
            residual = np.sqrt(np.mean((np.polyval(coeffs, x) - c60) ** 2))
            fit_quality = 1 - residual / (c60.std() + 1e-8)

            if a > 0 and fit_quality > 0.5 and c60[-1] > c60.mean():
                patterns.append({
                    "name": "圆弧底",
                    "direction": "bullish",
                    "confidence": int(60 + fit_quality * 15),
                    "desc": "价格走势呈U型圆弧底形态,中长期看涨信号",
                })
                score += 15
            elif a < 0 and fit_quality > 0.5 and c60[-1] < c60.mean():
                patterns.append({
                    "name": "圆弧顶",
                    "direction": "bearish",
                    "confidence": int(60 + fit_quality * 15),
                    "desc": "价格走势呈倒U型圆弧顶形态,中长期看跌信号",
                })
                score -= 15

        # --- 6. W底/M顶（大级别颈线突破）---
        if lookback >= 25:
            # W底：找两个相近低点，间隔>=10根
            win = 5
            local_lows = []
            for i in range(win, lookback - win):
                if l60[i] == min(l60[max(0,i-win):i+win+1]):
                    local_lows.append((i, l60[i]))
            # 找最相近的两个低点
            for i in range(len(local_lows)):
                for j in range(i + 1, len(local_lows)):
                    idx1, val1 = local_lows[i]
                    idx2, val2 = local_lows[j]
                    if idx2 - idx1 < 10:
                        continue
                    if abs(val1 - val2) / min(val1, val2) > 0.03:
                        continue  # 两底高度差>3%不算W底
                    # 颈线 = 两底之间的最高点
                    neckline = max(h60[idx1:idx2+1])
                    if c60[-1] > min(val1, val2) and neckline > min(val1, val2):
                        bo = DataBackend._confirm_breakout(c60, h60, l60, v60, neckline, "up")
                        target = neckline + (neckline - min(val1, val2))
                        p = {
                            "name": "W底", "direction": "bullish",
                            "confidence": 75 + (10 if bo["confirmed"] else 0),
                            "neckline": round(float(neckline), 2),
                            "target_price": round(float(target), 2),
                            "breakout": bo,
                            "desc": f"双底{min(val1,val2):.2f},颈线{neckline:.2f}",
                        }
                        if bo["confirmed"]:
                            score += 20
                            p["desc"] += f",已突破颈线(强度{bo['strength']},目标{target:.2f})"
                        elif c60[-1] > neckline * 0.97:
                            score += 8
                            p["desc"] += ",接近颈线"
                        patterns.append(p)
                        break
                else:
                    continue
                break

            # M顶：找两个相近高点
            local_highs = []
            for i in range(win, lookback - win):
                if h60[i] == max(h60[max(0,i-win):i+win+1]):
                    local_highs.append((i, h60[i]))
            for i in range(len(local_highs)):
                for j in range(i + 1, len(local_highs)):
                    idx1, val1 = local_highs[i]
                    idx2, val2 = local_highs[j]
                    if idx2 - idx1 < 10:
                        continue
                    if abs(val1 - val2) / max(val1, val2) > 0.03:
                        continue
                    neckline = min(l60[idx1:idx2+1])
                    if c60[-1] < max(val1, val2) and neckline < max(val1, val2):
                        bo = DataBackend._confirm_breakout(c60, h60, l60, v60, neckline, "down")
                        target = neckline - (max(val1, val2) - neckline)
                        p = {
                            "name": "M顶", "direction": "bearish",
                            "confidence": 75 + (10 if bo["confirmed"] else 0),
                            "neckline": round(float(neckline), 2),
                            "target_price": round(float(target), 2),
                            "breakout": bo,
                            "desc": f"双顶{max(val1,val2):.2f},颈线{neckline:.2f}",
                        }
                        if bo["confirmed"]:
                            score -= 20
                            p["desc"] += f",已跌破颈线(强度{bo['strength']},目标{target:.2f})"
                        elif c60[-1] < neckline * 1.03:
                            score -= 8
                            p["desc"] += ",接近颈线"
                        patterns.append(p)
                        break
                else:
                    continue
                break

        result["patterns"] = patterns
        result["chart_pattern_score"] = max(-40, min(40, score))
        if patterns:
            result["desc"] = "; ".join(f"{p['name']}({p['direction']},{p['confidence']}%)" for p in patterns)
        else:
            result["desc"] = "未检测到明显图形形态"
        return result

    @staticmethod
    def _detect_chanlun_signals(close, high, low, n: int) -> dict[str, Any]:
        """缠论（缠中说禅）核心信号检测：分型→笔→线段→中枢→买卖点。

        简化实现：
        1. 顶分型/底分型 检测（3根K线）
        2. 笔（Bi）构造：顶分型→底分型为下笔，底分型→顶分型为上笔
        3. 中枢（Hub/Pivot）：至少3笔重叠区间
        4. 三类买卖点判定

        返回 dict 包含分型列表、笔列表、中枢、买卖点信号。
        """
        import numpy as np

        result: dict[str, Any] = {
            "fractals": [],       # 分型列表 [{type, index, price}]
            "bi_list": [],        # 笔列表 [{dir, start_idx, end_idx, start_price, end_price}]
            "zhongshu": [],       # 中枢列表 [{high, low, start_idx, end_idx}]
            "buy_signals": [],    # 买点信号 [{type, price, desc}]
            "sell_signals": [],   # 卖点信号 [{type, price, desc}]
            "chanlun_score": 0,   # -50~+50
            "desc": "",
        }
        if n < 15:
            return result

        h = high.values.astype(float)
        l = low.values.astype(float)
        c = close.values.astype(float)

        # === 1. K线包含处理（合并包含关系）===
        # 方向：当前K线高点>=前K线高点为上，否则为下
        merged_h = [h[0]]
        merged_l = [l[0]]
        merged_idx = [0]
        for i in range(1, n):
            prev_h, prev_l = merged_h[-1], merged_l[-1]
            cur_h, cur_l = h[i], l[i]
            # 包含关系：一根K线完全包含另一根
            if (cur_h <= prev_h and cur_l >= prev_l) or (cur_h >= prev_h and cur_l <= prev_l):
                # 上升趋势取高高，下降趋势取低低
                if len(merged_h) >= 2 and merged_h[-1] > merged_h[-2]:
                    merged_h[-1] = max(prev_h, cur_h)
                    merged_l[-1] = max(prev_l, cur_l)
                else:
                    merged_h[-1] = min(prev_h, cur_h)
                    merged_l[-1] = min(prev_l, cur_l)
            else:
                merged_h.append(cur_h)
                merged_l.append(cur_l)
                merged_idx.append(i)

        mh = np.array(merged_h)
        ml = np.array(merged_l)
        mn = len(mh)

        # === 2. 分型检测（顶分型/底分型）===
        fractals = []  # (type, merged_index, original_index, price)
        for i in range(1, mn - 1):
            if mh[i] > mh[i-1] and mh[i] > mh[i+1]:
                # 顶分型
                fractals.append(("top", i, merged_idx[i] if i < len(merged_idx) else i, float(mh[i])))
            elif ml[i] < ml[i-1] and ml[i] < ml[i+1]:
                # 底分型
                fractals.append(("bot", i, merged_idx[i] if i < len(merged_idx) else i, float(ml[i])))

        result["fractals"] = [
            {"type": f[0], "index": f[2], "price": f[3]}
            for f in fractals[-20:]  # 只保留最近20个
        ]

        # === 3. 笔（Bi）构造 ===
        # 交替出现的顶底分型之间构成一笔，至少间隔4根合并K线
        bi_list = []
        if len(fractals) >= 2:
            last_fractal = fractals[0]
            for f in fractals[1:]:
                # 必须交替出现（顶→底 或 底→顶）
                if f[0] == last_fractal[0]:
                    # 同类型，取更极端的
                    if f[0] == "top" and f[3] > last_fractal[3]:
                        last_fractal = f
                    elif f[0] == "bot" and f[3] < last_fractal[3]:
                        last_fractal = f
                    continue
                # 检查间隔（至少4根合并K线）
                if abs(f[1] - last_fractal[1]) < 4:
                    continue
                direction = "up" if last_fractal[0] == "bot" else "down"
                bi_list.append({
                    "dir": direction,
                    "start_idx": last_fractal[2],
                    "end_idx": f[2],
                    "start_price": last_fractal[3],
                    "end_price": f[3],
                })
                last_fractal = f

        result["bi_list"] = bi_list[-15:]  # 最近15笔

        # === 4. 中枢（Zhongshu/Hub）检测 ===
        # 至少3笔有重叠区间构成中枢
        zhongshu_list = []
        if len(bi_list) >= 3:
            i = 0
            while i <= len(bi_list) - 3:
                # 取连续3笔的重叠区间
                ranges = []
                for j in range(i, min(i + 3, len(bi_list))):
                    bi = bi_list[j]
                    hi = max(bi["start_price"], bi["end_price"])
                    lo = min(bi["start_price"], bi["end_price"])
                    ranges.append((lo, hi))
                # 计算重叠区间
                zs_high = min(r[1] for r in ranges)
                zs_low = max(r[0] for r in ranges)
                if zs_high > zs_low:
                    # 中枢成立，尝试扩展
                    end_j = i + 3
                    while end_j < len(bi_list):
                        bi = bi_list[end_j]
                        hi = max(bi["start_price"], bi["end_price"])
                        lo = min(bi["start_price"], bi["end_price"])
                        new_high = min(zs_high, hi)
                        new_low = max(zs_low, lo)
                        if new_high > new_low:
                            end_j += 1
                        else:
                            break
                    zhongshu_list.append({
                        "high": round(zs_high, 2),
                        "low": round(zs_low, 2),
                        "start_idx": bi_list[i]["start_idx"],
                        "end_idx": bi_list[end_j - 1]["end_idx"],
                        "bi_count": end_j - i,
                    })
                    i = end_j
                else:
                    i += 1

        result["zhongshu"] = zhongshu_list[-5:]  # 最近5个中枢

        # === 5. 买卖点判定 ===
        chanlun_score = 0
        buy_signals = []
        sell_signals = []
        curr_price = float(c[-1])

        if zhongshu_list and bi_list:
            last_zs = zhongshu_list[-1]
            last_bi = bi_list[-1]

            # --- 一买：跌破中枢后出现底分型（背驰段结束）---
            if last_bi["dir"] == "down" and last_bi["end_price"] < last_zs["low"]:
                # 检查是否有底分型确认
                recent_bots = [f for f in fractals if f[0] == "bot" and f[2] >= last_bi["end_idx"] - 3]
                if recent_bots:
                    buy_signals.append({
                        "type": "一买",
                        "price": round(last_bi["end_price"], 2),
                        "desc": f"价格{last_bi['end_price']:.2f}跌破中枢下沿{last_zs['low']:.2f}后出现底分型,一类买点"
                    })
                    chanlun_score += 25

            # --- 二买：一买后回踩不破一买低点 ---
            if len(bi_list) >= 3:
                prev_bi = bi_list[-3]  # 可能的一买段
                mid_bi = bi_list[-2]   # 反弹段
                if (prev_bi["dir"] == "down" and mid_bi["dir"] == "up" and last_bi["dir"] == "down"
                    and last_bi["end_price"] > prev_bi["end_price"]):
                    buy_signals.append({
                        "type": "二买",
                        "price": round(last_bi["end_price"], 2),
                        "desc": f"回踩低点{last_bi['end_price']:.2f}未破前低{prev_bi['end_price']:.2f},二类买点"
                    })
                    chanlun_score += 20

            # --- 三买：回踩中枢上沿不跌入中枢 ---
            if (last_bi["dir"] == "down" and last_bi["end_price"] > last_zs["high"]
                and curr_price > last_zs["high"]):
                buy_signals.append({
                    "type": "三买",
                    "price": round(last_bi["end_price"], 2),
                    "desc": f"回踩{last_bi['end_price']:.2f}未跌入中枢上沿{last_zs['high']:.2f},三类买点"
                })
                chanlun_score += 18

            # --- 一卖：涨破中枢后出现顶分型 ---
            if last_bi["dir"] == "up" and last_bi["end_price"] > last_zs["high"]:
                recent_tops = [f for f in fractals if f[0] == "top" and f[2] >= last_bi["end_idx"] - 3]
                if recent_tops:
                    sell_signals.append({
                        "type": "一卖",
                        "price": round(last_bi["end_price"], 2),
                        "desc": f"价格{last_bi['end_price']:.2f}涨破中枢上沿{last_zs['high']:.2f}后出现顶分型,一类卖点"
                    })
                    chanlun_score -= 25

            # --- 二卖：一卖后反弹不破一卖高点 ---
            if len(bi_list) >= 3:
                prev_bi = bi_list[-3]
                mid_bi = bi_list[-2]
                if (prev_bi["dir"] == "up" and mid_bi["dir"] == "down" and last_bi["dir"] == "up"
                    and last_bi["end_price"] < prev_bi["end_price"]):
                    sell_signals.append({
                        "type": "二卖",
                        "price": round(last_bi["end_price"], 2),
                        "desc": f"反弹高点{last_bi['end_price']:.2f}未破前高{prev_bi['end_price']:.2f},二类卖点"
                    })
                    chanlun_score -= 20

            # --- 三卖：回升至中枢下沿受阻 ---
            if (last_bi["dir"] == "up" and last_bi["end_price"] < last_zs["low"]
                and curr_price < last_zs["low"]):
                sell_signals.append({
                    "type": "三卖",
                    "price": round(last_bi["end_price"], 2),
                    "desc": f"反弹{last_bi['end_price']:.2f}未回到中枢下沿{last_zs['low']:.2f},三类卖点"
                })
                chanlun_score -= 18

        # 额外：当前价格在中枢内 → 震荡
        if zhongshu_list:
            last_zs = zhongshu_list[-1]
            if last_zs["low"] <= curr_price <= last_zs["high"]:
                chanlun_score = int(chanlun_score * 0.5)  # 中枢内震荡，信号减半

        result["buy_signals"] = buy_signals
        result["sell_signals"] = sell_signals
        result["chanlun_score"] = max(-50, min(50, chanlun_score))

        descs = []
        if zhongshu_list:
            last_zs = zhongshu_list[-1]
            descs.append(f"最近中枢[{last_zs['low']:.2f}-{last_zs['high']:.2f}]({last_zs['bi_count']}笔)")
        if buy_signals:
            descs.append("买点:" + ",".join(s["type"] for s in buy_signals))
        if sell_signals:
            descs.append("卖点:" + ",".join(s["type"] for s in sell_signals))
        if bi_list:
            last_bi = bi_list[-1]
            descs.append(f"最近笔:{last_bi['dir']} {last_bi['start_price']:.2f}→{last_bi['end_price']:.2f}")
        result["desc"] = "; ".join(descs) if descs else "缠论信号不足"

        return result

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
