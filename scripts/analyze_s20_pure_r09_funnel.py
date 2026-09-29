#!/usr/bin/env python
"""Round 9: assemble the funnel and run the naive control.

Part A (dev test folds, 50% sample, learned heads available):
  F0 stage1 top-20% of the Top100 pool
  F1 naive amplitude cap : drop the pool's highest-natr 40%, re-take by stage1
  F2 learned cap         : drop the pool's worst 40% by whole-market C_down
  F3 direction head      : rank the pool by A_dir (chop-dropped training)
  F4 naive cap + A_dir
Part B (full stage1 universe, dev + confirm, no learned heads needed):
  naive amplitude cap with self-computed natr14 from daily bars, per-fold
  stability, and the exit-rule grid (U, D as parameters).
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

CAP = 0.4


def natr14() -> pd.DataFrame:
    fr = []
    for f in sorted((ROOT / "output/tushare_cache/daily").glob("*.parquet")):
        d = pd.read_parquet(f, columns=["ts_code", "trade_date", "high", "low", "close", "pre_close"])
        fr.append(d[d.ts_code.str.endswith((".SH", ".SZ"))])
    d = pd.concat(fr)
    d["trade_date"] = d["trade_date"].astype(str)
    d = d.sort_values(["ts_code", "trade_date"])
    tr = np.maximum(d.high - d.low, np.maximum((d.high - d.pre_close).abs(), (d.low - d.pre_close).abs()))
    d["natr14_self"] = (tr.groupby(d.ts_code).transform(lambda s: s.rolling(14, min_periods=10).mean()) / d.close)
    return d[["ts_code", "trade_date", "natr14_self"]]


def summarize(q: pd.DataFrame, name: str) -> dict:
    sh = q.st.value_counts(normalize=True)
    return {"funnel": name, "n": len(q), "pure_up": round(100 * sh.get("pure_up", 0), 1),
            "pure_down": round(100 * sh.get("pure_down", 0), 1), "chop": round(100 * sh.get("chop", 0), 1),
            "exit_mean": round(float(np.nanmean(q.exit)), 2), "exit_win": round(100 * (q.exit > 0).mean(), 1)}


def take(pool: pd.DataFrame, n_keep: pd.Series, by: str, ascending=False) -> pd.DataFrame:
    r = pool.groupby("trade_date")[by].rank(ascending=ascending, method="first")
    return pool[r <= pool.trade_date.map(n_keep)]


def cap(pool: pd.DataFrame, col: str, frac: float) -> pd.DataFrame:
    return pool[pool.groupby("trade_date")[col].rank(pct=True) <= 1 - frac]


def main() -> int:
    lines = []
    # ---------- Part A
    P = pd.read_parquet(OUT / "r05_preds.parquet")
    feat = pd.read_parquet(OUT / "feat_cache_bps5000.parquet", columns=["ts_code", "trade_date", "natr_14"])
    P = P.merge(feat, on=["ts_code", "trade_date"], how="left")
    pool = P[P.s1_rank <= 100].copy()
    n_keep = (pool.groupby("trade_date").size() * 0.2).round().clip(lower=1)
    rows = []
    for fo, g in [("all", pool), *pool.groupby("fold")]:
        nk = n_keep.loc[g.trade_date.unique()]
        for name, q in (("F0 stage1", take(g, nk, "stage1_score")),
                        ("F1 naive natr cap", take(cap(g, "natr_14", CAP), nk, "stage1_score")),
                        ("F2 C_down cap", take(cap(g, "C_down", CAP), nk, "stage1_score")),
                        ("F3 A_dir rank", take(g, nk, "A_dir")),
                        ("F4 natr cap + A_dir", take(cap(g, "natr_14", CAP), nk, "A_dir"))):
            rows.append({"fold": fo, **summarize(q, name)})
    lines.append("## Part A: dev test folds, 50% sample, pool = stage1 top100, output = 20% of pool\n"
                 + pd.DataFrame(rows).to_string(index=False))
    corr = pool.groupby("trade_date").apply(lambda g: g[["natr_14", "C_down", "A_dir"]].corr(method="spearman"))
    lines.append("\nmean daily Spearman inside pool:\n" + corr.groupby(level=1).mean().round(2).to_string())

    # ---------- Part B
    p = load().merge(natr14(), on=["ts_code", "trade_date"], how="left")
    p["period"] = np.where(p.trade_date <= "20260126", "dev", "confirm")
    p["rk"] = p.groupby("trade_date")["stage1_score"].rank(ascending=False, method="first")
    pool = p[p.rk <= 100].copy()
    rows = []
    for U, D in ((10, 8), (15, 10), (20, 10), (20, 15)):
        pool["exit"] = exit_return(pool, U, D)
        pool["st"] = states(pool, U, D).to_numpy()
        for per, g in pool.groupby("period"):
            for k in (20, 50):
                nk = pd.Series(k * 1.0, index=g.trade_date.unique())
                for c in (0.0, 0.2, 0.4, 0.6):
                    base = g if c == 0 else cap(g, "natr14_self", c)
                    rows.append({"U": U, "D": D, "period": per, "TopK": k, "natr_cap": c,
                                 **summarize(take(base, nk, "stage1_score"), "")})
    t = pd.DataFrame(rows).drop(columns="funnel")
    lines.append("\n## Part B: full stage1 universe, pool top100 -> drop highest-natr c -> stage1 TopK\n" + t.to_string(index=False))

    # monthly stability for U15/D10, Top20, cap 0 vs 0.4
    pool["exit"] = exit_return(pool, 15, 10)
    pool["st"] = states(pool, 15, 10).to_numpy()
    pool["month"] = pool.trade_date.str[:6]
    m = []
    for mo, g in pool.groupby("month"):
        nk = pd.Series(20.0, index=g.trade_date.unique())
        a, b = take(g, nk, "stage1_score"), take(cap(g, "natr14_self", 0.4), nk, "stage1_score")
        m.append({"month": mo, "top20_exit": round(float(np.nanmean(a.exit)), 2),
                  "cap40_exit": round(float(np.nanmean(b.exit)), 2),
                  "top20_down": round(100 * (a.st == "pure_down").mean(), 1),
                  "cap40_down": round(100 * (b.st == "pure_down").mean(), 1)})
    mm = pd.DataFrame(m)
    lines.append("\n## monthly, U15/D10 Top20: stage1 vs natr-cap40\n" + mm.to_string(index=False))
    lines.append(f"months cap40 better exit: {(mm.cap40_exit > mm.top20_exit).sum()}/{len(mm)}; "
                 f"lower pure_down: {(mm.cap40_down < mm.top20_down).sum()}/{len(mm)}")
    text = "\n".join(lines)
    (OUT / "r09_funnel.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
