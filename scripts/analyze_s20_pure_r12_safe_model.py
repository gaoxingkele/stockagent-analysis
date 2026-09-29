#!/usr/bin/env python
"""Round 12: train for the user's actual objective - "safe and up".

Label (a = +5% lower band edge, D = -10% crash line, 20 sessions, T+1 open):
  success = win (+a before -D) or small win (neither, day-20 close > 0)
  bad     = -D before +a
  crash15 = 20-session max drawdown <= -15%
Heads (same 166 portable factors, purged walk-forward S20_V2_FOLDS, 50% sample):
  S   success vs rest
  Bd  bad vs rest
  Cr  crash15 vs rest
Compared on the daily Top-N of each ranking against:
  stage1 (amplitude selector), naive low-natr, universe average.
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from stockagent_analysis.s20 import purged_walk_forward_masks  # noqa: E402
from train_s20_v2_multitarget import S20_V2_FOLDS  # noqa: E402
from analyze_s20_pure_r04_direction import PARAMS, load  # noqa: E402

OUT = ROOT / "output/experiments/s20_pure_20260928"
A, B, D, COST = 5, 15, 10, 0.3


def fit(f, t, feats, yf, yt):
    return lgb.train(PARAMS, lgb.Dataset(f[feats], label=yf), num_boost_round=600,
                     valid_sets=[lgb.Dataset(t[feats], label=yt)],
                     callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])


def metrics(q):
    cls, ret = q[f"cls_a{A}_d{D}"], q[f"ret_a{A}_b{B}_d{D}"] - COST
    return {"n": len(q), "success": round(100 * cls.isin([1, 3]).mean(), 1), "bad": round(100 * (cls == 2).mean(), 1),
            "crash15": round(100 * (q.maxdd20 <= -15).mean(), 1), "band_mean": round(float(ret.mean()), 2),
            "worst_month": round(float(ret.groupby(q.trade_date.str[:6]).mean().min()), 2),
            "median_natr_pct": round(float(q.natr_pct.median()), 2)}


def main() -> int:
    data, feats = load(15, 10, 5000)
    band = pd.read_parquet(OUT / "band_panel.parquet",
                           columns=["ts_code", "trade_date", "maxdd20", f"cls_a{A}_d{D}", f"ret_a{A}_b{B}_d{D}"])
    data = data.merge(band, on=["ts_code", "trade_date"], how="inner")
    cls = data[f"cls_a{A}_d{D}"]
    data["y_succ"], data["y_bad"], data["y_crash"] = cls.isin([1, 3]), cls.eq(2), data.maxdd20 <= -15
    preds = []
    for fold in S20_V2_FOLDS:
        m = purged_walk_forward_masks(data["trade_date"], data["horizon_end_date"], fold)
        f, t, te = data[m["fit"]], data[m["tune"]], data[m["test"]].copy()
        its = []
        for name, y in (("S", "y_succ"), ("Bd", "y_bad"), ("Cr", "y_crash")):
            mdl = fit(f, t, feats, f[y], t[y])
            te[name] = mdl.predict(te[feats])
            its.append(mdl.best_iteration)
        te["fold"] = fold.name
        preds.append(te[["ts_code", "trade_date", "fold", "natr_14", "S", "Bd", "Cr", "maxdd20",
                         f"cls_a{A}_d{D}", f"ret_a{A}_b{B}_d{D}"]])
        print(f"{fold.name}: fit {len(f):,} test {len(te):,} iters S/Bd/Cr {its}", flush=True)
    P = pd.concat(preds, ignore_index=True)
    s1 = pd.read_parquet(ROOT / "output/experiments/s20_20r_stage1_forward_outcomes/scored_with_outcomes.parquet",
                         columns=["ts_code", "trade_date", "stage1_score"])
    P = P.merge(s1, on=["ts_code", "trade_date"], how="left")
    P = P[P.stage1_score.notna()].copy()      # same universe as stage1 (ST excluded, scored days)
    P["natr_pct"] = P.groupby("trade_date")["natr_14"].rank(pct=True)
    P["S_minus_Bd"] = P.S - P.Bd
    P["S_x_notCr"] = P.S * (1 - P.Cr)
    P.to_parquet(OUT / "r12_preds.parquet", index=False)

    lines = [f"success = +{A}% before -{D}% or positive day-20 close; band exit a{A}/b{B}; 50% sample; dev test folds"]
    rows = [{"ranking": "universe", "N": "-", **metrics(P)}]
    for N in (10, 20, 40):   # 50% sample: Top-N here ~ Top-2N of the full market
        for name, col, asc in (("stage1", "stage1_score", False), ("low natr (naive)", "natr_14", True),
                               ("S", "S", False), ("-Bd", "Bd", True), ("-Cr", "Cr", True),
                               ("S-Bd", "S_minus_Bd", False), ("S*(1-Cr)", "S_x_notCr", False)):
            r = P.groupby("trade_date")[col].rank(ascending=asc, method="first")
            rows.append({"ranking": name, "N": N, **metrics(P[r <= N])})
    t = pd.DataFrame(rows)
    lines.append(t.to_string(index=False))
    lines.append("\n## per fold, N=20")
    rows = []
    for fo, g in P.groupby("fold"):
        rows.append({"fold": fo, "ranking": "universe", **metrics(g)})
        for name, col, asc in (("stage1", "stage1_score", False), ("low natr", "natr_14", True),
                               ("S", "S", False), ("S-Bd", "S_minus_Bd", False), ("S*(1-Cr)", "S_x_notCr", False)):
            r = g.groupby("trade_date")[col].rank(ascending=asc, method="first")
            rows.append({"fold": fo, "ranking": name, **metrics(g[r <= 20])})
    lines.append(pd.DataFrame(rows).to_string(index=False))
    text = "\n".join(lines)
    (OUT / "r12_safe_model.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
