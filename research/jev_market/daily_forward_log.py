#!/usr/bin/env python
"""Phase 2: forward-only Jev log of market-wide sell-off risk (no look-ahead possible).

Run once per trading day after the CCTV evening news (~19:40 Beijing time):
    python research/jev_market/daily_forward_log.py            # today
    python research/jev_market/daily_forward_log.py 20260929   # a specific evening (same day only)

1. Market snapshot after the close from the Sina A-share spot table (no token needed),
   appended to output/jev_market/live_market_days.csv (seeded from the Tushare daily cache).
2. That evening's CCTV news via AKShare.
3. Jev is asked the same frozen question as the phase-1 probe, in two conditions
   (A_real: date + snapshot + news; D_numeric: snapshot only), and the answers are
   appended with a UTC timestamp to output/jev_market/forward_log.jsonl.
Outcomes are attached later by research/jev_market/evaluate_forward_log.py.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_market_crash import FEATS, ROOT, ask, api_key, cctv, iso, market_snapshot, snapshot_dict  # noqa: E402

OUT = ROOT / "output/jev_market"
LIVE = OUT / "live_market_days.csv"
LOG = OUT / "forward_log.jsonl"
QUESTION_VERSION = "selloff5_v1"


def spot_row(date: str) -> dict:
    import akshare as ak
    df = ak.stock_zh_a_spot()
    code = df["代码"].astype(str)
    df = df[code.str.startswith(("sh6", "sz0", "sz3"))].copy()
    pct = pd.to_numeric(df["涨跌幅"], errors="coerce")
    board20 = df["代码"].str[2:5].isin(["300", "301", "688", "689"])
    return {"date": date, "median_ret": float(pct.median()), "up_share": float((pct > 0).mean()),
            "limit_up": int((pct >= np.where(board20, 19.5, 9.5)).sum()),
            "limit_down": int((pct <= np.where(board20, -19.5, -9.5)).sum()),
            "amount": float(pd.to_numeric(df["成交额"], errors="coerce").sum()) / 1000.0,  # 元 -> 千元 (Tushare unit)
            "source": "sina_spot"}


def live_table() -> pd.DataFrame:
    if LIVE.exists():
        return pd.read_csv(LIVE, dtype={"date": str})
    seed = market_snapshot()[["date", "median_ret", "up_share", "limit_up", "limit_down", "amount"]]
    seed["source"] = "tushare_cache"
    seed.to_csv(LIVE, index=False)
    return seed


def main() -> int:
    date = sys.argv[1] if len(sys.argv) > 1 else dt.date.today().strftime("%Y%m%d")
    if date != dt.date.today().strftime("%Y%m%d"):
        raise SystemExit("forward log only records the current evening (spot data is live)")
    if LOG.exists() and any(json.loads(l)["date"] == date for l in LOG.read_text(encoding="utf-8").splitlines() if l):
        print(f"{date} already logged"); return 0
    live = live_table()
    if date not in set(live.date):
        live = pd.concat([live, pd.DataFrame([spot_row(date)])], ignore_index=True)
        live.to_csv(LIVE, index=False)
    m = live.sort_values("date").reset_index(drop=True)
    m["median_ret_5d"] = m.median_ret.rolling(5).sum()
    m["median_ret_20d"] = m.median_ret.rolling(20).sum()
    m["limit_down_5d"] = m.limit_down.rolling(5).sum()
    m["amount_z60"] = (m.amount - m.amount.rolling(60).mean()) / m.amount.rolling(60).std()
    row = m[m.date == date].iloc[0]
    snap, news = snapshot_dict(row), cctv(date)
    if not news:
        raise SystemExit("CCTV news not available yet; run after ~19:40")
    key = api_key()
    rec = {"date": date, "logged_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "question_version": QUESTION_VERSION, "snapshot": snap,
           "news_sha256": hashlib.sha256(json.dumps(news, ensure_ascii=False).encode()).hexdigest()}
    for cond, state in (("A_real", {"as_of": iso(date), "market_after_close": snap, "cctv_evening_news": news}),
                        ("D_numeric", {"market_after_close": snap})):
        r = ask(state, key)
        rec[cond] = r.get("p")
        rec["model"] = r.get("model", rec.get("model"))
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps({k: rec[k] for k in ("date", "A_real", "D_numeric", "model")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
