#!/usr/bin/env python
"""Intraday structure features for S20 signal days, from 15-minute bars (30 and 60 minutes are built here).

Bars are stamped at their end time, 17 per session: 09:30 (opening auction), 09:45 .. 11:30, 13:15 .. 15:00.
  60 minutes: 10:30 (09:30..10:30), 11:30, 14:00 (13:15..14:00), 15:00
  30 minutes: 10:00 (09:30..10:00), 10:30, 11:00, 11:30, 13:30 (13:15, 13:30), 14:00, 14:30, 15:00
The 16 features are the ones registered in wiki/2026-10-07_intraday-structure-s20.md before any result
was seen. Window = the signal day and the 4 sessions before it. Windows that contain an ex-rights day
(daily pre_close differs from the previous close) are dropped, since minute bars are unadjusted.

Output: output/experiments/intraday_s20_20261007/features.parquet  (ts_code, trade_date, 16 features)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MIN15 = ROOT / "output/tushare_cache/min15"
OUT = ROOT / "output/experiments/intraday_s20_20261007"
FEATURES = ["eff15_5d", "eff60_5d", "jump_share_5d", "overnight_share_5d", "last_hour_ret_t", "pm_minus_am_5d",
            "above_vwap_share_t", "close_vs_vwap_t", "updown_vol_ratio_5d", "last_hour_vol_share_5d",
            "macd_resonance", "macd60_hist_slope", "rsi60_t", "bars_since_golden60", "dd_from_5d_high", "higher_lows_60m"]
H60 = {"09:30": 1, "09:45": 1, "10:00": 1, "10:15": 1, "10:30": 1, "10:45": 2, "11:00": 2, "11:15": 2, "11:30": 2,
       "13:15": 3, "13:30": 3, "13:45": 3, "14:00": 3, "14:15": 4, "14:30": 4, "14:45": 4, "15:00": 4}
H30 = {"09:30": 1, "09:45": 1, "10:00": 1, "10:15": 2, "10:30": 2, "10:45": 3, "11:00": 3, "11:15": 4, "11:30": 4,
       "13:15": 5, "13:30": 5, "13:45": 6, "14:00": 6, "14:15": 7, "14:30": 7, "14:45": 8, "15:00": 8}


def resample(b: pd.DataFrame, buckets: dict) -> pd.DataFrame:
    x = b.assign(bk=b.hm.map(buckets))
    return x.groupby(["trade_date", "bk"], sort=True).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                                          close=("close", "last"), vol=("vol", "sum")).reset_index()


def macd_hist(close: pd.Series) -> pd.Series:
    dif = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    return dif - dif.ewm(span=9, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def stock_features(code: str, signal_days: list[str], atr: pd.Series, exdays: set[str]) -> pd.DataFrame:
    b = pd.read_parquet(MIN15 / f"{code}.parquet")
    if b.empty:
        return pd.DataFrame()
    b["trade_date"] = b.trade_time.str[:10].str.replace("-", "")
    b["hm"] = b.trade_time.str[11:16]
    b = b[b.hm.isin(H60)].sort_values("trade_time").reset_index(drop=True)
    days = sorted(b.trade_date.unique())
    pos = {d: i for i, d in enumerate(days)}
    # per-session aggregates on 15 minutes
    b["ret"] = b.close.pct_change()
    first = b.trade_date.ne(b.trade_date.shift())
    b["overnight"] = b.ret.where(first)                 # previous 15:00 close -> today's auction bar close
    b["intra"] = b.ret.where(~first)
    b["pv"] = b.close * b.vol
    b["vwap"] = b.groupby("trade_date").pv.cumsum() / b.groupby("trade_date").vol.cumsum().replace(0, np.nan)
    g = b.groupby("trade_date")
    day = pd.DataFrame({
        "close": g.close.last(), "high": g.high.max(),
        "abs_intra": g.intra.apply(lambda s: s.abs().sum()), "max_abs_intra": g.intra.apply(lambda s: s.abs().max()),
        "abs_overnight": g.overnight.apply(lambda s: s.abs().sum()),
        "am_ret": g.apply(lambda x: x.loc[x.hm == "11:30", "close"].iloc[0] / x.open.iloc[0] - 1 if (x.hm == "11:30").any() else np.nan),
        "last_close_1400": g.apply(lambda x: x.loc[x.hm == "14:00", "close"].iloc[0] if (x.hm == "14:00").any() else np.nan),
        "close_1130": g.apply(lambda x: x.loc[x.hm == "11:30", "close"].iloc[0] if (x.hm == "11:30").any() else np.nan),
        "vol": g.vol.sum(), "vol_last_hour": g.apply(lambda x: x.loc[x.hm >= "14:15", "vol"].sum()),
        "up_vol": g.apply(lambda x: x.loc[x.intra > 0, "vol"].sum()), "dn_vol": g.apply(lambda x: x.loc[x.intra < 0, "vol"].sum()),
        "above_vwap": g.apply(lambda x: (x.close > x.vwap).mean()), "vwap": g.vwap.last(),
    })
    day["pm_ret"] = day.close / day.close_1130 - 1
    h60 = resample(b, H60)
    h30 = resample(b, H30)
    for frame in (b, h30, h60):
        frame["mh"] = macd_hist(frame.close)
    h60["rsi"] = rsi(h60.close)
    golden = (h60.mh > 0) & (h60.mh.shift() <= 0)
    idx = np.arange(len(h60))
    last_g = pd.Series(np.where(golden, idx, np.nan)).ffill().to_numpy()
    h60["since_golden"] = np.minimum(idx - last_g, 40)
    h60.loc[np.isnan(last_g), "since_golden"] = 40
    h60["higher_low"] = (h60.low > h60.low.shift()).astype(float)
    rows = []
    for t in signal_days:
        if t not in pos or pos[t] < 4:
            continue
        win = days[pos[t] - 4: pos[t] + 1]
        if exdays & set(win[1:]):
            continue
        w15 = b[b.trade_date.isin(win)]
        w60 = h60[h60.trade_date.isin(win)]
        d = day.loc[win]
        net15 = w15.close.iloc[-1] - w15.close.iloc[0]
        path15 = w15.close.diff().abs().sum()
        net60 = w60.close.iloc[-1] - w60.close.iloc[0]
        path60 = w60.close.diff().abs().sum()
        intra_abs = w15.intra.abs().sum()
        last60 = h60[h60.trade_date <= t].tail(8)
        e15, e30, e60 = b[b.trade_date == t].mh.iloc[-1], h30[h30.trade_date == t].mh.iloc[-1], h60[h60.trade_date == t].mh.iloc[-1]
        h60_t = h60[h60.trade_date <= t]
        a = atr.get(t, np.nan)
        rows.append({
            "ts_code": code, "trade_date": t,
            "eff15_5d": net15 / path15 if path15 > 0 else np.nan,
            "eff60_5d": net60 / path60 if path60 > 0 else np.nan,
            "jump_share_5d": w15.intra.abs().max() / intra_abs if intra_abs > 0 else np.nan,
            "overnight_share_5d": d.abs_overnight.iloc[1:].sum() / (d.abs_overnight.iloc[1:].sum() + d.abs_intra.sum()),
            "last_hour_ret_t": d.close.iloc[-1] / d.last_close_1400.iloc[-1] - 1,
            "pm_minus_am_5d": (d.pm_ret - d.am_ret).mean(),
            "above_vwap_share_t": d.above_vwap.iloc[-1],
            "close_vs_vwap_t": d.close.iloc[-1] / d.vwap.iloc[-1] - 1,
            "updown_vol_ratio_5d": d.up_vol.sum() / d.dn_vol.sum() if d.dn_vol.sum() > 0 else np.nan,
            "last_hour_vol_share_5d": d.vol_last_hour.sum() / d.vol.sum() if d.vol.sum() > 0 else np.nan,
            "macd_resonance": float(int(e15 > 0) + int(e30 > 0) + int(e60 > 0)),
            "macd60_hist_slope": (h60_t.mh.iloc[-1] - h60_t.mh.iloc[-5]) / d.close.iloc[-1] if len(h60_t) >= 5 else np.nan,
            "rsi60_t": h60_t.rsi.iloc[-1],
            "bars_since_golden60": h60_t.since_golden.iloc[-1],
            "dd_from_5d_high": (d.close.iloc[-1] - w15.high.max()) / a if a and a > 0 else np.nan,
            "higher_lows_60m": last60.higher_low.iloc[1:].sum(),
        })
    return pd.DataFrame(rows)


def _run(job: tuple) -> pd.DataFrame:
    return stock_features(*job)


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from price_panel import adjusted_daily  # noqa: E402
    from stockagent_analysis.s20_pure import PureConfig, select  # noqa: E402
    OUT.mkdir(parents=True, exist_ok=True)
    fr = pd.read_parquet(ROOT / "output/experiments/s20_frozen_rescore_20261005/frozen_scores.parquet")
    fr = fr[fr.trade_date <= "20260728"]
    surv = select(fr, PureConfig(top_k=100), "U15D10")[["ts_code", "trade_date"]]
    codes = sorted(set(surv.ts_code) & {p.stem for p in MIN15.glob("*.parquet")})
    raw = []
    for f in sorted((ROOT / "output/tushare_cache/daily").glob("*.parquet")):
        if "20250101" <= f.stem <= "20260805":
            x = pd.read_parquet(f, columns=["ts_code", "trade_date", "close", "pre_close", "high", "low"])
            raw.append(x[x.ts_code.isin(codes)])
    raw = pd.concat(raw)
    raw["trade_date"] = raw.trade_date.astype(str)
    raw = raw.sort_values(["ts_code", "trade_date"])
    raw["ex"] = (raw.pre_close - raw.groupby("ts_code").close.shift()).abs() > 0.011
    tr = np.maximum(raw.high - raw.low, np.maximum((raw.high - raw.pre_close).abs(), (raw.low - raw.pre_close).abs()))
    raw["atr"] = tr.groupby(raw.ts_code).transform(lambda s: s.rolling(14, min_periods=10).mean())
    from concurrent.futures import ProcessPoolExecutor
    jobs = []
    for code in codes:
        r = raw[raw.ts_code == code]
        jobs.append((code, sorted(surv.loc[surv.ts_code == code, "trade_date"]), r.set_index("trade_date").atr,
                     set(r.loc[r.ex, "trade_date"])))
    parts = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        for n, f in enumerate(pool.map(_run, jobs, chunksize=8), 1):
            if len(f):
                parts.append(f)
            if n % 200 == 0:
                print(f"  {n}/{len(codes)} stocks", flush=True)
    feats = pd.concat(parts, ignore_index=True)
    feats.to_parquet(OUT / "features.parquet", index=False)
    print(f"features: {len(feats):,} stock-days, {feats.ts_code.nunique()} stocks, coverage of survivors {len(feats) / len(surv):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
