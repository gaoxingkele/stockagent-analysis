#!/usr/bin/env python
"""Did the winners in the S20 / R20 top-100 pools have a bottom pattern shortly before the signal day?

  python scripts/analyze_success_precursors.py

Reverse question (user 2026-10-12): among past pool names that worked, how many showed some double-bottom
kind of pattern within the previous k sessions (k = 5, 10, 20, the signal day included)? The answer only
means something next to the same share among the names that failed, so both are reported, plus the
success rate with vs without the pattern (same-day paired difference, Newey-West(19) t).
Outcome: list-basis U15D10 from next-day open: success = +15% touched before -10% within 20 sessions,
failure = -10% first (ties count as failure), the rest are time exits.
Patterns: B1-B7 from scripts/bottom_patterns.py (B1 = Chan MACD-area divergence, i.e. 背驰), plus
  B8 Bollinger bottom turn: a close back above the lower band (20, 2) after >= 1 close below it in the
     previous 5 sessions, on an up day
  B9 MACD DIF bottom divergence (背离): two confirmed pivot lows 10-60 sessions apart, the later lower in
     price but higher in DIF, DIF < 0 at the first
  B10 the WorkBuddy W rule, broad version (scripts/compare_workbuddy_double_bottom.py)
  ANY any of the above.
Pools and periods: S20 frozen stage1 top 100 (S1 2025-03-03..08-29, S2 2026-01-27..08-05); R20 production
r20_pred top 100 (01-27..04-13, 04-14..08-05). Signal days >= 2026-08-06 are not read. Periods already used:
evidence tier reused_holdout, descriptive.
Output: output/experiments/success_precursors_20261012/{report.txt, flags.parquet}
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
from analyze_bottom_patterns import pools  # noqa: E402
from analyze_r20_s20_nested import tstat  # noqa: E402

OUT = ROOT / "output/experiments/success_precursors_20261012"
SEGS = {"S20": {"S1 2025-03..08": ("20250303", "20250829"), "S2 2026-01-27..08-05": ("20260127", "20260805")},
        "R20": {"R20 in-sample 01-27..04-13": ("20260127", "20260413"), "R20 OOS 04-14..08-05": ("20260414", "20260805")}}
NAMES = {"B1": "缠论背驰(MACD面积)", "B9": "MACD DIF底背离", "B4": "RSI底背离", "B2c": "W底突破", "B2f": "W底形成",
         "B10": "W底(WorkBuddy宽口径)", "B3": "双插针+MA5拐头", "B8": "布林带底部回转", "B5": "KDJ超卖金叉",
         "B6": "吞没/晨星", "B7": "缩量后放量", "ANY": "任一形态"}
KS = (5, 10, 20)
LINES: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    LINES.append(s)


def extra_stock(l: np.ndarray, c: np.ndarray) -> dict[str, np.ndarray]:
    n = len(c)
    b8, b9 = np.zeros(n, bool), np.zeros(n, bool)
    s = pd.Series(c)
    lower = (s.rolling(20).mean() - 2 * s.rolling(20).std(ddof=0)).to_numpy()
    below = c < lower
    for D in range(25, n):
        if c[D] > lower[D] and c[D] > c[D - 1] and below[D - 5:D].any() and not below[D]:
            b8[D] = True
    dif = (s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()).to_numpy()
    ls = pd.Series(l)
    piv = np.where((ls == ls.rolling(7, center=True, min_periods=7).min()).to_numpy())[0]
    for D in range(80, n):
        conf = piv[piv <= D - 3]
        p2c = conf[conf >= D - 14]
        if not len(p2c):
            continue
        P2 = p2c[-1]
        for P1 in conf[(conf <= P2 - 10) & (conf >= P2 - 60)]:
            if l[P2] < l[P1] and dif[P2] > dif[P1] and dif[P1] < 0 and dif[D] > dif[P2]:
                b9[D] = True
                break
    return {"B8": b8, "B9": b9}


def _chunk(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for code, g in frame.groupby("ts_code", sort=False):
        r = extra_stock(g.low.to_numpy(float), g.close.to_numpy(float))
        parts.append(pd.DataFrame({"ts_code": code, "trade_date": g.trade_date.to_numpy(), **r}))
    return pd.concat(parts, ignore_index=True)


def flags_panel() -> pd.DataFrame:
    from price_panel import adjusted_daily
    px = adjusted_daily("20240801", "20260805")
    codes = px.ts_code.unique()
    with ProcessPoolExecutor(8) as ex:
        ex_ = pd.concat(ex.map(_chunk, [px[px.ts_code.isin(set(codes[i::8]))] for i in range(8)]), ignore_index=True)
    pp = pd.read_parquet(ROOT / "output/experiments/bottom_patterns_20261012/patterns.parquet",
                         columns=["ts_code", "trade_date", "B1", "B2c", "B2f", "B3", "B4", "B5", "B6", "B7"])
    f = ex_.merge(pp, on=["ts_code", "trade_date"], how="left")
    wb = pd.read_parquet(ROOT / "output/experiments/workbuddy_compare_20261012/wb_hits.parquet", columns=["ts_code", "trade_date"])
    f["B10"] = pd.MultiIndex.from_frame(f[["ts_code", "trade_date"]]).isin(pd.MultiIndex.from_frame(wb))
    f = f[f.trade_date >= "20250102"].sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    pats = [k for k in NAMES if k != "ANY"]
    f[pats] = f[pats].fillna(False).astype(bool)
    f["ANY"] = f[pats].any(axis=1)
    g = f.groupby("ts_code", sort=False)
    for k in KS:   # pattern seen on any of the last k sessions, signal day included
        roll = g[list(NAMES)].rolling(k, min_periods=1).max().reset_index(level=0, drop=True)
        for p in NAMES:
            f[f"{p}_{k}"] = roll[p].astype(bool).to_numpy()
    return f


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "flags.parquet"
    if cache.exists():
        f = pd.read_parquet(cache)
    else:
        f = flags_panel()
        f.to_parquet(cache, index=False)
    o = pd.read_parquet(ROOT / "output/experiments/s20_alt_designs_20261011/paths.parquet",
                        columns=["ts_code", "trade_date", "ret_base", "up15_day", "dn10_day"])
    o["success"] = (o.up15_day > 0) & ((o.dn10_day == 0) | (o.up15_day < o.dn10_day))
    o["failure"] = (o.dn10_day > 0) & ((o.up15_day == 0) | (o.dn10_day <= o.up15_day))
    P = pools()
    say("pattern seen on any of the last k sessions (signal day included); success = +15% before -10% in 20 sessions")
    say("market base rates of the 20-session window flag, 2025-03..2026-08-05: "
        + ", ".join(f"{p} {100 * f.loc[f.trade_date.between('20250303', '20260805'), f'{p}_20'].mean():.0f}%" for p in NAMES))
    for pname, segs in SEGS.items():
        base = P[pname].merge(o, on=["ts_code", "trade_date"]).merge(f, on=["ts_code", "trade_date"], how="left")
        for sname, (a, b) in segs.items():
            g = base[base.trade_date.between(a, b)].copy()
            S, F = g[g.success], g[g.failure]
            say(f"\n==== {pname} top 100, {sname}: {len(g):,} cases on {g.trade_date.nunique()} days; success {100 * g.success.mean():.1f}%, "
                f"failure {100 * g.failure.mean():.1f}%, time exit {100 * (1 - g.success.mean() - g.failure.mean()):.1f}%; per trade {g.ret_base.mean():+.2f}")
            for k in KS:
                say(f"-- k = {k}: pattern | share among successes | among failures | all | success rate with / without | paired diff (t) | per trade with / without")
                for p in NAMES:
                    col = f"{p}_{k}"
                    x = g[col].fillna(False).astype(bool)
                    w, wo = g[x], g[~x]
                    d = (w.groupby("trade_date").success.mean() - wo.groupby("trade_date").success.mean()).dropna() * 100
                    say(f"   {NAMES[p]:18s} | {100 * S[col].fillna(False).mean():5.1f}% | {100 * F[col].fillna(False).mean():5.1f}% | {100 * x.mean():5.1f}% | "
                        f"{100 * w.success.mean() if len(w) else float('nan'):5.1f}% / {100 * wo.success.mean():5.1f}% | "
                        f"{d.mean() if len(d) else float('nan'):+5.1f} ({tstat(d) if len(d) > 5 else float('nan'):+.1f}) | "
                        f"{w.ret_base.mean() if len(w) else float('nan'):+.2f} / {wo.ret_base.mean():+.2f}")
    (OUT / "report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
