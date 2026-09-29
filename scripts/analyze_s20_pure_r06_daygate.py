#!/usr/bin/env python
"""Round 6: is the day-level part of "pure down" visible before the trade?

Target per signal day: stage1 Top20 exit-rule mean (U15/D10) and pure_down share.
Ex-ante day features only (known at the signal-day close).
Reports Spearman IC on dev days and confirm days, with the effective number of
independent 20-session windows, and the "abstain on worst-predicted X% of days"
book effect. Descriptive: the confirm window is already consumed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_s20_pure_r01_states import OUT, load, states  # noqa: E402
from analyze_s20_pure_r03_asymmetry import exit_return  # noqa: E402


def breadth() -> pd.DataFrame:
    rows = []
    files = sorted((ROOT / "output/tushare_cache/daily").glob("*.parquet"))
    for f in files:
        d = pd.read_parquet(f, columns=["ts_code", "pct_chg", "close", "pre_close"])
        d = d[d.ts_code.str.endswith((".SH", ".SZ"))]
        rows.append({"trade_date": f.stem, "adv_share": (d.pct_chg > 0).mean(),
                     "limit_up_share": (d.pct_chg >= 9.8).mean(), "limit_dn_share": (d.pct_chg <= -9.8).mean(),
                     "xs_ret_mean": d.pct_chg.clip(-20, 20).mean(), "xs_ret_disp": d.pct_chg.clip(-20, 20).std()})
    b = pd.DataFrame(rows).sort_values("trade_date")
    for c in ("adv_share", "xs_ret_mean", "limit_dn_share", "xs_ret_disp"):
        b[f"{c}_5d"] = b[c].rolling(5).mean()
    return b


def main() -> int:
    p = load()
    p["rk"] = p.groupby("trade_date")["stage1_score"].rank(ascending=False, method="first")
    p["exit"] = exit_return(p, 15, 10)
    p["st"] = states(p, 15, 10).to_numpy()
    top = p[p.rk <= 20]
    day = top.groupby("trade_date").agg(exit_mean=("exit", "mean"),
                                        pure_down=("st", lambda s: (s == "pure_down").mean()),
                                        s1_top_mean=("stage1_score", "mean"))
    day["univ_exit"] = p.groupby("trade_date")["exit"].mean()
    day["s1_top_vs_univ"] = day["s1_top_mean"] - p.groupby("trade_date")["stage1_score"].mean()
    reg = pd.read_parquet(ROOT / "output/regimes/daily_regime.parquet")
    reg["trade_date"] = reg["trade_date"].astype(str)
    ext = pd.read_parquet(ROOT / "output/regime_extra/regime_extra.parquet")
    ext["trade_date"] = ext["trade_date"].astype(str)
    day = day.reset_index().merge(reg.drop(columns=["regime"]), on="trade_date", how="left") \
        .merge(ext, on="trade_date", how="left").merge(breadth(), on="trade_date", how="left")
    day["period"] = np.where(day.trade_date <= "20260126", "dev", "confirm")
    feats = [c for c in day.columns if c not in ("trade_date", "exit_mean", "pure_down", "univ_exit", "period")]

    lines = [f"days: dev {int((day.period=='dev').sum())}, confirm {int((day.period=='confirm').sum())}; "
             f"independent 20-session windows ~ {len(day)//20}"]
    lines.append(f"Top20 exit mean: day-to-day sd {day.exit_mean.std():.2f}pp; corr with universe exit "
                 f"{day[['exit_mean','univ_exit']].corr().iloc[0,1]:.2f}")
    rows = []
    for c in feats:
        r = {"feature": c}
        for per in ("dev", "confirm"):
            q = day[day.period == per]
            r[f"IC_exit_{per}"] = round(q[c].corr(q.exit_mean, method="spearman"), 3)
            r[f"IC_down_{per}"] = round(q[c].corr(q.pure_down, method="spearman"), 3)
        # sign-stable across the two dev halves and confirm?
        d = day[day.period == "dev"]
        h1, h2 = d.iloc[: len(d) // 2], d.iloc[len(d) // 2:]
        r["IC_exit_dev_h1"] = round(h1[c].corr(h1.exit_mean, method="spearman"), 3)
        r["IC_exit_dev_h2"] = round(h2[c].corr(h2.exit_mean, method="spearman"), 3)
        rows.append(r)
    t = pd.DataFrame(rows)
    t["stable"] = (np.sign(t.IC_exit_dev_h1) == np.sign(t.IC_exit_dev_h2)) & \
                  (np.sign(t.IC_exit_dev_h1) == np.sign(t.IC_exit_confirm)) & \
                  (t[["IC_exit_dev_h1", "IC_exit_dev_h2", "IC_exit_confirm"]].abs().min(axis=1) >= 0.1)
    lines.append(t.sort_values("IC_exit_dev", key=abs, ascending=False).to_string(index=False))

    # abstention effect of each sign-stable feature: skip the worst 30% predicted days (threshold from dev)
    lines.append("\n## abstain on the 30% of days a feature ranks worst (threshold chosen on dev, applied to confirm)")
    for c in t.loc[t.stable, "feature"]:
        sgn = np.sign(t.set_index("feature").loc[c, "IC_exit_dev"])
        d = day[day.period == "dev"]
        thr = (sgn * d[c]).quantile(0.3)
        for per in ("dev", "confirm"):
            q = day[day.period == per]
            keep = (sgn * q[c]) > thr
            lines.append(f"{c:22s} {per:7s}: all-days exit {q.exit_mean.mean():+.2f}  kept {keep.mean()*100:4.0f}% of days "
                         f"-> {q.loc[keep,'exit_mean'].mean():+.2f}; skipped days {q.loc[~keep,'exit_mean'].mean():+.2f}")
    text = "\n".join(lines)
    (OUT / "r06_daygate.txt").write_text(text, encoding="utf-8")
    day.to_parquet(OUT / "r06_day_panel.parquet", index=False)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
