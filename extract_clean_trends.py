#!/usr/bin/env python3
"""提取"干净上涨"样本.

定义:
  对每个 (ts_code, trade_date) 作为潜在买点 t0:
    - 取 t1..t20 的 OHLC
    - max_gain_20 = max(high[1..20]) / open[1] - 1
    - max_dd_20  = min(low[1..20])  / open[1] - 1   (回撤, 负数)
  干净上涨标准:
    max_gain_20 >= 0.20  AND  max_dd_20 >= -0.03

输出:
  output/clean_trends/clean_trends.parquet
    columns: ts_code, trade_date, max_gain_20, max_dd_20, r20, total_mv, pe, industry
  output/clean_trends/summary.json   (按 mv × pe × ETF 分桶汇总)

数据源: TDX 本地 vipdoc 二进制 (D:/tdx/vipdoc/)
"""
from __future__ import annotations

import json
import logging
import os
import struct
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("extract_clean_trends")

ROOT = Path(__file__).resolve().parent
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
OUT_DIR     = ROOT / "output" / "clean_trends"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 干净走势阈值
MAX_GAIN_THRESH = 0.20    # 20 天内最高涨幅 >= 20%
MAX_DD_THRESH   = -0.03   # 20 天内最大回撤 >= -3%
LOOKAHEAD       = 20

# 数据范围
START = "20240101"
END   = "20260126"

TDX_DIR = os.getenv("TDX_DIR", "D:/tdx")


def _read_tdx_day_full(market: str, code: str) -> pd.DataFrame | None:
    """读 TDX 日线, 返回 (date, open, high, low, close, volume) 全量."""
    path = Path(TDX_DIR) / "vipdoc" / market / "lday" / f"{market}{code}.day"
    if not path.is_file():
        return None
    try:
        data = path.read_bytes()
    except Exception:
        return None
    n = len(data) // 32
    if n == 0:
        return None
    records = []
    for i in range(n):
        offset = i * 32
        fields = struct.unpack_from("<8I", data, offset)
        date_int = fields[0]
        try:
            d = dt.date(date_int // 10000, (date_int % 10000) // 100, date_int % 100)
        except ValueError:
            continue
        records.append({
            "date": d.strftime("%Y%m%d"),
            "open":  fields[1] / 100.0,
            "high":  fields[2] / 100.0,
            "low":   fields[3] / 100.0,
            "close": fields[4] / 100.0,
        })
    return pd.DataFrame(records) if records else None


def _guess_market(code: str) -> str:
    if code.startswith(("8", "4", "9")): return "bj"
    if code.startswith(("5", "6")):       return "sh"
    return "sz"


def _ts_code_to_code_market(ts: str) -> tuple[str, str]:
    """ '600519.SH' → ('600519', 'sh') """
    if "." in ts:
        c, ex = ts.split(".")
        return c, ex.lower()
    return ts, _guess_market(ts)


def compute_clean_signals(daily: pd.DataFrame) -> pd.DataFrame:
    """对单股 daily 计算每日是否为干净走势起点.

    daily 需要按日期升序, 含 open/high/low/close.
    返回 DataFrame: date, max_gain_20, max_dd_20, is_clean
    """
    n = len(daily)
    if n < LOOKAHEAD + 1:
        return pd.DataFrame()
    op  = daily["open"].values
    hi  = daily["high"].values
    lo  = daily["low"].values
    cl  = daily["close"].values
    dates = daily["date"].values

    out = []
    for i in range(n - LOOKAHEAD):
        # 假设 t0 收盘后决策, t1 开盘买入
        # i = t0 (信号生成日), i+1 = t1 (买入日, 用 open)
        # i+1 ... i+LOOKAHEAD = 持有期
        entry_open = op[i + 1]
        if entry_open <= 0: continue
        future_hi = hi[i + 1: i + 1 + LOOKAHEAD]
        future_lo = lo[i + 1: i + 1 + LOOKAHEAD]
        max_gain = float(future_hi.max() / entry_open - 1)
        max_dd   = float(future_lo.min() / entry_open - 1)
        out.append({
            "date": dates[i],
            "entry_open": entry_open,
            "max_gain_20": round(max_gain, 4),
            "max_dd_20": round(max_dd, 4),
            "r20_close": round(cl[i + LOOKAHEAD] / entry_open - 1, 4),
            "is_clean": (max_gain >= MAX_GAIN_THRESH) and (max_dd >= MAX_DD_THRESH),
        })
    return pd.DataFrame(out)


def bucket_mv(mv_wan):
    if mv_wan is None or pd.isna(mv_wan): return None
    bil = mv_wan / 1e4
    if bil < 50: return "20-50亿"
    if bil < 100: return "50-100亿"
    if bil < 300: return "100-300亿"
    if bil < 1000: return "300-1000亿"
    return "1000亿+"


def bucket_pe(pe):
    if pe is None or pd.isna(pe): return None
    if pe < 0: return "亏损"
    if pe < 15: return "0-15"
    if pe < 30: return "15-30"
    if pe < 50: return "30-50"
    if pe < 100: return "50-100"
    return "100+"


def main():
    t0 = time.time()
    log.info("加载 parquet meta (用于 mv/pe/industry)...")
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p, columns=["ts_code", "trade_date", "total_mv",
                                          "pe", "pe_ttm", "industry"])
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= START) & (df["trade_date"] <= END)]
        parts.append(df)
    meta = pd.concat(parts, ignore_index=True)
    log.info("Meta: %d 行, %d 股票", len(meta), meta["ts_code"].nunique())

    # ETF 持仓信息 (可选)
    etf_holders = set()
    try:
        etf_json = ROOT / "output" / "etf_tracker" / "stock_to_etfs.json"
        if etf_json.exists():
            data = json.loads(etf_json.read_text(encoding="utf-8"))
            etf_holders = set(data.keys())
            log.info("ETF 持仓集合: %d 股", len(etf_holders))
    except Exception:
        pass

    # 逐股算干净走势
    log.info("逐股扫描 TDX 日线 (LOOKAHEAD=%d, gain>=%.0f%%, dd>=%.0f%%)...",
             LOOKAHEAD, MAX_GAIN_THRESH * 100, MAX_DD_THRESH * 100)
    all_signals = []
    ts_codes = meta["ts_code"].unique()
    found = 0
    skipped = 0
    for i, ts in enumerate(ts_codes):
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            log.info("  [%d/%d] 已扫 %ds, 干净信号 %d, 跳过 %d",
                     i + 1, len(ts_codes), int(elapsed), found, skipped)
        code, market = _ts_code_to_code_market(ts)
        daily = _read_tdx_day_full(market, code)
        if daily is None or len(daily) < LOOKAHEAD + 5:
            skipped += 1
            continue
        daily = daily[(daily["date"] >= START) & (daily["date"] <= END)].sort_values("date").reset_index(drop=True)
        if len(daily) < LOOKAHEAD + 5:
            skipped += 1
            continue
        sigs = compute_clean_signals(daily)
        if sigs.empty:
            continue
        sigs["ts_code"] = ts
        clean = sigs[sigs["is_clean"]]
        if not clean.empty:
            all_signals.append(clean)
            found += len(clean)

    if not all_signals:
        log.error("没有找到任何干净走势!")
        return

    clean_df = pd.concat(all_signals, ignore_index=True)
    log.info("总干净信号: %d 条 / %d 股票",
             len(clean_df), clean_df["ts_code"].nunique())

    # 合并 meta 信息
    clean_df = clean_df.rename(columns={"date": "trade_date"})
    clean_df = clean_df.merge(meta, on=["ts_code", "trade_date"], how="left")
    clean_df["mv_seg"] = clean_df["total_mv"].apply(bucket_mv)
    pe_col = clean_df["pe_ttm"].fillna(clean_df["pe"])
    clean_df["pe_seg"] = pe_col.apply(bucket_pe)
    clean_df["etf_held"] = clean_df["ts_code"].isin(etf_holders)

    # 输出
    out_path = OUT_DIR / "clean_trends.parquet"
    clean_df.to_parquet(out_path, index=False)
    log.info("写出 %s", out_path)

    # 汇总: 按 mv × pe × etf 分桶
    log.info("\n=== 按 mv × pe 分桶分布 ===")
    pivot = clean_df.pivot_table(index="mv_seg", columns="pe_seg",
                                  values="ts_code", aggfunc="count", fill_value=0)
    print(pivot.to_string())

    log.info("\n=== 按 ETF 持仓 vs 非持仓 ===")
    by_etf = clean_df.groupby("etf_held").agg(
        n=("ts_code", "count"),
        avg_max_gain=("max_gain_20", "mean"),
        avg_max_dd=("max_dd_20", "mean"),
        avg_r20=("r20_close", "mean"),
    ).round(4)
    print(by_etf.to_string())

    log.info("\n=== 按行业 Top 15 ===")
    by_ind = clean_df.groupby("industry").size().sort_values(ascending=False).head(15)
    print(by_ind.to_string())

    # 元数据
    summary = {
        "n_total_clean":  int(len(clean_df)),
        "n_stocks":       int(clean_df["ts_code"].nunique()),
        "n_dates":        int(clean_df["trade_date"].nunique()),
        "thresholds":     {"max_gain": MAX_GAIN_THRESH, "max_dd": MAX_DD_THRESH,
                            "lookahead": LOOKAHEAD},
        "data_period":    [START, END],
        "by_mv":          clean_df["mv_seg"].value_counts().to_dict(),
        "by_pe":          clean_df["pe_seg"].value_counts().to_dict(),
        "by_etf":         clean_df["etf_held"].value_counts().to_dict(),
        "by_industry_top10": clean_df["industry"].value_counts().head(10).to_dict(),
        "elapsed_sec":    round(time.time() - t0, 1),
    }
    Path(OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    log.info("汇总写入 summary.json (%.1fs)", time.time() - t0)


if __name__ == "__main__":
    main()
