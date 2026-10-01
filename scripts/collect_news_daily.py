#!/usr/bin/env python
"""Collect stock news, announcements and market flashes with fetch timestamps (forward archive).

Public sources (AKShare / Eastmoney, CLS), no LLM here: raw text is stored first,
labels are added later by a separate tagging step, so collection never waits on
an LLM provider.

    python scripts/collect_news_daily.py                 # today: candidates' news + market flashes + notices
    python scripts/collect_news_daily.py --notices-from 20240101 --notices-to 20260930   # backfill notice titles

Candidates = latest S20 lists (v1 + safe) + latest R20 Pool A + stage1 pool, capped.
Output (one file per collection day, append + de-dup by link/title):
  output/news/stock_news/<YYYYMMDD>.parquet
  output/news/market_flash/<YYYYMMDD>.parquet
  output/news/notices/<YYYYMMDD>.parquet       (all-market announcement titles of that date)
"""
from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path

import pandas as pd

pd.set_option("future.infer_string", False)   # akshare regexes break on pandas' arrow strings
import akshare as ak  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NEWS = ROOT / "output/news"
SHADOW = ROOT / "output/experiments/s20_pure_v1_shadow"
R20 = ROOT / "output/r20_history/days"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def append(path: Path, df: pd.DataFrame, keys: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True)
    df = df.drop_duplicates(subset=keys, keep="first")
    df.to_parquet(path, index=False)
    return len(df)


def candidates(cap: int = 200) -> list[str]:
    codes: list[str] = []
    raw = SHADOW / "daily_lists_raw.csv"
    if raw.exists():
        L = pd.read_csv(raw, dtype={"trade_date": str, "ts_code": str})
        codes += L[L.trade_date == L.trade_date.max()].ts_code.tolist()
    days = sorted(R20.glob("*.parquet"))
    if days:
        A = pd.read_parquet(days[-1])
        codes += A[A["list"] == "pool_a"].ts_code.astype(str).tolist()
    seen, out = set(), []
    for c in codes:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out[:cap]


def stock_news(day: str, codes: list[str]) -> int:
    rows = []
    for i, ts_code in enumerate(codes, 1):
        for attempt in range(3):
            try:
                df = ak.stock_news_em(symbol=ts_code.split(".")[0])
                df["ts_code"] = ts_code
                df["fetched_at_utc"] = now()
                rows.append(df)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    print(f"  news {ts_code} failed: {str(exc)[:60]}", flush=True)
                time.sleep(2)
        time.sleep(0.3)
    if not rows:
        return 0
    return append(NEWS / "stock_news" / f"{day}.parquet", pd.concat(rows, ignore_index=True), ["ts_code", "新闻链接"])


def market_flash(day: str) -> int:
    parts = []
    for name, fn in (("cls", lambda: ak.stock_info_global_cls(symbol="全部")), ("em", ak.stock_info_global_em)):
        try:
            df = fn().astype(str)
            df["source"] = name
            df["fetched_at_utc"] = now()
            parts.append(df)
        except Exception as exc:  # noqa: BLE001
            print(f"  flash {name} failed: {str(exc)[:60]}", flush=True)
    if not parts:
        return 0
    df = pd.concat(parts, ignore_index=True)
    key = "内容" if "内容" in df.columns else "标题"
    df[key] = df[key].fillna(df.get("摘要", ""))
    return append(NEWS / "market_flash" / f"{day}.parquet", df, ["source", key])


def notices(day: str) -> int:
    for attempt in range(3):
        try:
            df = ak.stock_notice_report(symbol="全部", date=day)
            break
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                print(f"  notices {day} failed: {str(exc)[:60]}", flush=True)
                return 0
            time.sleep(3)
    if df is None or df.empty:
        return 0
    df = df.astype(str)
    df["fetched_at_utc"] = now()
    return append(NEWS / "notices" / f"{day}.parquet", df, ["代码", "公告标题"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notices-from")
    ap.add_argument("--notices-to")
    a = ap.parse_args()
    if a.notices_from:
        days = pd.bdate_range(a.notices_from, a.notices_to).strftime("%Y%m%d")
        for d in days:
            if (NEWS / "notices" / f"{d}.parquet").exists():
                continue
            print(f"notices {d}: {notices(d)}", flush=True)
            time.sleep(0.5)
        return 0
    day = dt.date.today().strftime("%Y%m%d")
    codes = candidates()
    print(f"{day}: candidates {len(codes)}", flush=True)
    print(f"  stock news rows (cumulative today): {stock_news(day, codes)}", flush=True)
    print(f"  market flash rows: {market_flash(day)}", flush=True)
    print(f"  notices rows: {notices(day)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
