#!/usr/bin/env python
"""Test the 16 pre-registered intraday structure features on the S20 funnel survivors.

Registration: wiki/2026-10-07_intraday-structure-s20.md (written before any minute-bar result was seen).
For each feature and each day, the survivors are split into thirds; high third minus low third, same day,
for the 20-session +15/-10 exit return and the stop-first share; Newey-West(19) t. Development
2025-03-03..2026-01-26 and confirmation 2026-01-27..2026-07-28 are reported separately.
Pass: same sign in both, |t| >= 2 in both, and |t| >= 3.0 on the pooled days (32 comparisons).
Also: 5-day clean start-up / start-down (method C), and the mean daily rank correlation of each feature
with NATR(14) and the past 5-day return, to catch volatility or momentum under another name.
Output: output/experiments/intraday_s20_20261007/report.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from experiment_cross_filter_v2 import nw_se  # noqa: E402
from intraday_bars import FEATURES, OUT  # noqa: E402
from price_panel import adjusted_daily  # noqa: E402
from s20_pure_history import outcomes  # noqa: E402
from stockagent_analysis.pump_labels import clean_start_labels  # noqa: E402
from stockagent_analysis.s20_pure import PureConfig, select  # noqa: E402

DEV_END = "20260126"


def tstat(d: pd.Series) -> float:
    d = d.dropna()
    se = nw_se(d.to_numpy()) if len(d) > 20 else np.nan
    return d.mean() / se if se and se > 0 else np.nan


def main() -> int:
    feats = pd.read_parquet(OUT / "features.parquet")
    fr = pd.read_parquet(ROOT / "output/experiments/s20_frozen_rescore_20261005/frozen_scores.parquet")
    fr = fr[fr.trade_date <= "20260728"]
    surv = select(fr, PureConfig(top_k=100), "U15D10")[["ts_code", "trade_date", "list_rank", "natr14"]]
    out = outcomes()
    out = out[out.trade_date <= "20260728"][["ts_code", "trade_date", "ret_v1", "state"]]
    px = adjusted_daily("20250101", "20260805", set(surv.ts_code))
    lab = clean_start_labels(px[["ts_code", "trade_date", "high", "low", "close", "pre_close"]])
    px["past5"] = px.groupby("ts_code").close.pct_change(5)
    s = (surv.merge(feats, on=["ts_code", "trade_date"]).merge(out, on=["ts_code", "trade_date"])
         .merge(lab[["ts_code", "trade_date", "label"]], on=["ts_code", "trade_date"], how="left")
         .merge(px[["ts_code", "trade_date", "past5"]], on=["ts_code", "trade_date"], how="left"))
    s["stop"] = (s.state == "pure_down").astype(float)
    s["up"] = s.state.isin(["pure_up", "dirty_up"]).astype(float)
    s["c_up"] = np.where(s.label.isna(), np.nan, (s.label == 2).astype(float))
    s["c_dn"] = np.where(s.label.isna(), np.nan, (s.label == 1).astype(float))
    s["seg"] = np.where(s.trade_date <= DEV_END, "dev", "conf")
    lines = [f"S20 survivors with intraday features: {len(s):,} rows, {s.trade_date.nunique()} days "
             f"(dev {s[s.seg == 'dev'].trade_date.nunique()}, conf {s[s.seg == 'conf'].trade_date.nunique()}); "
             f"mean per trade {s.ret_v1.mean():+.2f}, stop-first {100 * s.stop.mean():.1f}%", "",
             "feature                 | rho NATR  rho past5 | per trade high-low: dev (t)  conf (t)  all (t) | stop-first high-low (pts): dev (t) conf (t) all (t) | 5d clean-up / clean-down high-low (pts) | verdict"]
    passed = []
    for f in FEATURES:
        x = s.dropna(subset=[f]).copy()
        x["t"] = x.groupby("trade_date")[f].transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=False) if v.nunique() >= 3 and len(v) >= 9 else np.nan)
        rho_n = x.groupby("trade_date").apply(lambda g: g[f].corr(g.natr14, method="spearman")).mean()
        rho_p = x.groupby("trade_date").apply(lambda g: g[f].corr(g.past5, method="spearman")).mean()
        cell = {}
        for m in ("ret_v1", "stop", "c_up", "c_dn"):
            d = (x[x.t == 2].groupby("trade_date")[m].mean() - x[x.t == 0].groupby("trade_date")[m].mean())
            seg = x.groupby("trade_date").seg.first()
            cell[m] = {k: (d[seg.reindex(d.index) == k] if k != "all" else d) for k in ("dev", "conf", "all")}
        r, st = cell["ret_v1"], cell["stop"]
        ok = all(np.sign(r["dev"].mean()) == np.sign(r["conf"].mean()) and abs(tstat(r[k])) >= 2 for k in ("dev", "conf")) and abs(tstat(r["all"])) >= 3.0
        ok_s = all(np.sign(st["dev"].mean()) == np.sign(st["conf"].mean()) and abs(tstat(st[k])) >= 2 for k in ("dev", "conf")) and abs(tstat(st["all"])) >= 3.0
        verdict = ("PASS ret" if ok else "") + (" PASS stop" if ok_s else "")
        if ok or ok_s:
            passed.append(f)
        lines.append(f"{f:23s} | {rho_n:+.2f}     {rho_p:+.2f}     | "
                     + "  ".join(f"{r[k].mean():+5.2f}({tstat(r[k]):+4.1f})" for k in ("dev", "conf", "all")) + " | "
                     + "  ".join(f"{100 * st[k].mean():+5.1f}({tstat(st[k]):+4.1f})" for k in ("dev", "conf", "all")) + " | "
                     + f"{100 * cell['c_up']['all'].mean():+4.1f} / {100 * cell['c_dn']['all'].mean():+4.1f} | {verdict or '-'}")
    lines += ["", f"passed: {passed if passed else 'none'}"]
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
