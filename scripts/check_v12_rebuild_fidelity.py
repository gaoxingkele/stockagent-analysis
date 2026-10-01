#!/usr/bin/env python
"""Does the rebuilt feature store reproduce production? Three checks.

(a) factor_lab: rebuilt *_ext_rebuild vs production *_ext_review on shared dates,
    per-column share of rows that differ (rel. tol 1e-6).
(b) aux features: recompute amount / mf1 / mfk / pyramid / v7 for a stock sample
    and compare with the production day files on shared dates.
(c) end to end: V12Scorer.score_market on days production published a list, then
    Pool A (R20 target rule, >= 20260828) or the V7c main pool (earlier days),
    compared with dashboard_*/poolA_system.csv (Jaccard and exact-set share).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

FL_EXT = ROOT / "output/factor_lab_3y/factor_groups_extension"


def diff_share(a: pd.DataFrame, b: pd.DataFrame, keys=("ts_code", "trade_date")) -> pd.Series:
    j = a.merge(b, on=list(keys), suffixes=("", "_p"))
    out = {}
    for c in a.columns:
        if c in keys or f"{c}_p" not in j:
            continue
        x = pd.to_numeric(j[c], errors="coerce")
        y = pd.to_numeric(j[f"{c}_p"], errors="coerce")
        both_nan = x.isna() & y.isna()
        close = np.isclose(x, y, rtol=1e-6, atol=1e-9, equal_nan=False)
        out[c] = float((~(both_nan | close)).mean())
    s = pd.Series(out).sort_values(ascending=False)
    s.attrs["rows"] = len(j)
    return s


def check_factor_lab(lines):
    reb, prod = [], []
    for p in sorted(FL_EXT.glob("group_*_ext_rebuild.parquet")):
        q = p.with_name(p.name.replace("_ext_rebuild", "_ext_review"))
        if q.exists():
            a, b = pd.read_parquet(p), pd.read_parquet(q)
            for x in (a, b):
                x["trade_date"] = x.trade_date.astype(str)
            reb.append(a[a.trade_date.isin(set(b.trade_date))])
            prod.append(b)
    s = diff_share(pd.concat(reb), pd.concat(prod))
    lines.append(f"(a) factor_lab rebuilt vs production on shared dates: {s.attrs['rows']:,} rows, "
                 f"{len(s)} columns; columns with any difference: {int((s > 0).sum())}")
    lines.append("    worst columns: " + ", ".join(f"{k} {v:.1%}" for k, v in s.head(8).items()))


def check_aux(lines, n_stocks=60):
    import rebuild_v12_features as R
    days = sorted(p.stem for p in (ROOT / "output/tushare_cache/daily").glob("*.parquet"))
    start, end = days[-5], days[-1]
    daily = R.load_daily(end)
    codes = sorted(daily)[:: max(1, len(daily) // n_stocks)][:n_stocks]
    for mod in (R._am, R._mfk, R._pyr, R._v7):
        mod.END = end
    for key, (out_dir, name) in R.AUX.items():
        prod = [pd.read_parquet(p) for p in sorted(out_dir.glob(f"{name}_ext_*.parquet"))
                if p.stem.endswith(end[-4:])]
        if not prod:
            continue
        prod = pd.concat(prod)
        prod["trade_date"] = prod.trade_date.astype(str)
        rows = []
        for ts in codes:
            d = daily[ts]
            try:
                if key == "amount":
                    x = R._am.compute_amount_features(list(zip(d.trade_date, d.open, d.high, d.low, d.close, d.vol)), start, end)
                    x = x.assign(ts_code=ts)
                else:
                    mf = pd.read_parquet(R.MF_CACHE / f"{ts}.parquet")
                    mf["trade_date"] = mf.trade_date.astype(str)
                    close = list(zip(d.trade_date, d.close))
                    x = {"mf1": lambda: R.compute_mf_v1(mf), "mfk": lambda: R._mfk.compute_mfk(mf, close),
                         "pyr": lambda: R._pyr.compute_pyramid_v2(mf), "v7": lambda: R._v7.compute_v7(mf, close)}[key]()
                x["trade_date"] = x.trade_date.astype(str)
                rows.append(x[(x.trade_date >= start) & (x.trade_date <= end)])
            except Exception:  # noqa: BLE001
                pass
        s = diff_share(pd.concat(rows), prod)
        lines.append(f"(b) {key:6s}: {s.attrs['rows']} rows, differing columns {int((s > 0).sum())}/{len(s)}"
                     + ("" if s.max() == 0 else f"; worst {s.index[0]} {s.iloc[0]:.1%}"))


def check_end_to_end(lines, max_days: int | None = None):
    import daily_dashboard as dash
    from stockagent_analysis.v12_scoring import V12Scorer
    scorer = V12Scorer.get(ROOT)
    pubs = sorted((ROOT / "output/daily_pick").glob("dashboard_*/poolA_system.csv"))
    have = {p.stem for p in (ROOT / "output/tushare_cache/daily").glob("*.parquet")}
    rows = []
    for p in pubs[-max_days:] if max_days else pubs:
        d = p.parent.name.split("_")[1]
        if d not in have:
            continue
        pub = pd.read_csv(p, dtype={"ts_code": str})
        try:
            df = scorer.score_market(d)
        except Exception as exc:  # noqa: BLE001
            rows.append({"date": d, "error": str(exc)[:80]})
            continue
        df["ratio"] = df["pump_score"] / (df["pump_down_score"] + 0.01)
        if "a_recommend" in pub.columns:
            basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet")[["ts_code", "industry"]]
            if "industry" not in df.columns:
                df = df.merge(basic.drop_duplicates("ts_code"), on="ts_code", how="left")
            mine, rule = set(dash._build_pool_a(df)[0].ts_code), "pool_a_r20"
        else:
            mine, rule = set(df.loc[df.v7c_recommend.fillna(False).astype(bool), "ts_code"]), "v7c_main"
        theirs = set(pub.ts_code)
        inter = len(mine & theirs)
        rows.append({"date": d, "rule": rule, "published": len(theirs), "replayed": len(mine), "common": inter,
                     "jaccard": round(inter / max(len(mine | theirs), 1), 3),
                     "published_recalled": round(inter / max(len(theirs), 1), 3)})
        print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    lines.append("(c) end-to-end list reproduction\n" + t.to_string(index=False))
    if "jaccard" in t:
        for rule, g in t.dropna(subset=["jaccard"]).groupby("rule"):
            lines.append(f"    {rule}: mean Jaccard {g.jaccard.mean():.3f}, published names recalled "
                         f"{g.published_recalled.mean():.1%} over {len(g)} days")


def main() -> int:
    lines = []
    check_factor_lab(lines)
    check_aux(lines)
    check_end_to_end(lines)
    text = "\n".join(lines)
    out = ROOT / "output/r20_history/rebuild_fidelity.txt"
    out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
