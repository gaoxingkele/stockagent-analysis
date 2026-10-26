#!/usr/bin/env python
"""Why did pool A grow from 8 to 42 names on 2026-10-23? Decompose the r20 prediction of the jumped names.

  python scripts/diagnose_pool_a_jump.py [--prev 20261022] [--day 20261023]

Descriptive diagnosis (user request 2026-10-26), not an experiment.
 1. feature-group staleness: does each group file hold a row for each day (else the scorer silently falls back to the file's latest day)
 2. the jumped set J = r20_pred >= 25 on `day` and < 25 on `prev`; LightGBM pred_contrib (raw margin space) per feature on both days
 3. per feature: mean change in contribution over J, change in the feature value (as cross-sectional percentile) over J and over all names
 4. same decomposition for all other names (is the shift specific to J or market-wide?)
Output: output/experiments/poolA_jump_20261023/report.txt, contrib.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
from export_pump_history import stale_groups  # noqa: E402
from stockagent_analysis.v12_scoring import V12Scorer  # noqa: E402

OUT = ROOT / "output/experiments/poolA_jump_20261023"
LINES: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    LINES.append(s)


def feats(sc, day: str):
    booster, meta = sc._models["r20_v16_all"]
    df = sc.load_factors_for_date(day).copy()
    df["industry_id"] = df["industry"].fillna("unknown").map(lambda x: meta.get("industry_map", {}).get(str(x), -1)) if "industry" in df else -1
    cols = meta["feature_cols"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    X = df[cols].astype(float)
    X.index = df["ts_code"].astype(str).values
    return X, booster, cols


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev", default="20261020")
    ap.add_argument("--day", default="20261023")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sc = V12Scorer.get(ROOT)
    sc._load_models()
    for d in (a.prev, a.day):
        say(f"stale feature groups on {d}: {stale_groups(sc, d) or 'none'}")
    X0, booster, cols = feats(sc, a.prev)
    X1, _, _ = feats(sc, a.day)
    p0 = pd.Series(booster.predict(X0), index=X0.index)
    p1 = pd.Series(booster.predict(X1), index=X1.index)
    say(f"model r20: {len(cols)} features; names {len(p0)} / {len(p1)}")
    common = p0.index.intersection(p1.index)
    J = [c for c in common if p1[c] >= 25 and p0[c] < 25]
    R = [c for c in common if c not in set(J)]
    say(f"jumped set (>=25 on {a.day}, <25 on {a.prev}): {len(J)}; scale of prediction: raw units as pool A uses (percent)")
    say(f"mean r20_pred change: J {np.mean([p1[c] - p0[c] for c in J]):+.2f}; all others {np.mean([p1[c] - p0[c] for c in R]):+.3f}; median others {np.median([p1[c] - p0[c] for c in R]):+.3f}")
    C0 = pd.DataFrame(booster.predict(X0, pred_contrib=True)[:, :-1], index=X0.index, columns=cols)
    C1 = pd.DataFrame(booster.predict(X1, pred_contrib=True)[:, :-1], index=X1.index, columns=cols)
    P0, P1 = X0.rank(pct=True), X1.rank(pct=True)
    X0i, X1i = X0, X1
    dC = (C1.loc[J] - C0.loc[J]).mean()
    dCr = (C1.loc[R] - C0.loc[R]).mean()
    dV_J = (P1.loc[J] - P0.loc[J]).mean()
    dV_R = (P1.loc[R] - P0.loc[R]).mean()
    chg = ((X1i.loc[J] - X0i.loc[J]).abs() > 1e-12).mean()
    t = pd.DataFrame({"d_contrib_J": dC, "d_contrib_others": dCr, "d_pct_J": dV_J, "d_pct_others": dV_R,
                      "share_J_value_changed": chg, "J_value_prev_med": X0i.loc[J].median(), "J_value_now_med": X1i.loc[J].median()})
    t["abs"] = t.d_contrib_J.abs()
    t = t.sort_values("abs", ascending=False)
    t.drop(columns="abs").to_csv(OUT / "contrib.csv", encoding="utf-8-sig")
    say(f"sum of contribution changes over J (margin units): {dC.sum():+.3f}; others {dCr.sum():+.3f}")
    say(f"top 15 features by |mean change in contribution| over J:")
    say(t.drop(columns="abs").head(15).round(3).to_string())
    top = list(t.index[:15])
    say(f"\nfeatures with value change in <50% of J names (stale?): {[c for c in cols if chg[c] < 0.5][:40]}")
    say(f"share of J covered by top5 / top15 contributions: {dC.loc[top[:5]].sum() / dC.sum():.2f} / {dC.loc[top].sum() / dC.sum():.2f}")
    # counterfactual: swap only the market-level columns (identical for every stock on a day)
    mk = [c for c in cols if c.startswith(("mkt_", "hs300_", "zz500_", "cyb_", "regime"))]
    say(f"\nmarket-level columns ({len(mk)}): {mk}")
    Xa = X1.copy(); Xa[mk] = X0[mk].iloc[0].values          # day's stock data, prev day's market state
    Xb = X0.copy(); Xb[mk] = X1[mk].iloc[0].values          # prev stock data, day's market state
    say(f"names with r20_pred >= 25: {a.prev} actual {int((p0 >= 25).sum())}; {a.day} actual {int((p1 >= 25).sum())}; "
        f"{a.day} stocks + {a.prev} market state {int((booster.predict(Xa) >= 25).sum())}; "
        f"{a.prev} stocks + {a.day} market state {int((booster.predict(Xb) >= 25).sum())}")
    for nm, c in (("mkt_vol_ratio only", ["mkt_vol_ratio"]), ("mkt_rsi14 only", ["mkt_rsi14"]), ("mkt_vol_ratio+mkt_rsi14+mkt_ret_5d", ["mkt_vol_ratio", "mkt_rsi14", "mkt_ret_5d"])):
        Xc = X1.copy(); Xc[c] = X0[c].iloc[0].values
        say(f"  {a.day} stocks with {nm} set back to {a.prev}: {int((booster.predict(Xc) >= 25).sum())} names >= 25")
    (OUT / "report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
