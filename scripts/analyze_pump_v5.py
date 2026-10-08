#!/usr/bin/env python
"""Evaluate pump v5 (shape edition) against v3c, then on the S20 funnel.

Test window 2026-01-27..2026-07-28 (never used to train or stop the model). v3c is out of its own
training window there too; its validation (early stopping) ran to 2026-05-22, so the test is split at
that date: "early" overlaps v3c's validation, "late" is out-of-sample for both models.

  1. Label prediction, full market: for each model and each label (method C clean start-up/down, and
     the v3c label "+10% without -5%"), daily one-vs-rest AUC and daily top-5% precision.
  2. Is v5 just momentum / just stage1? rank correlation of v5 P(up) with the past 5- and 20-day return
     and, inside the S20 survivors, with stage1.
  3. S20 survivors (frozen funnel, cut 40%): P(up) and P(down) terciles -> 5-day clean labels and the
     20-day +15/-10 outcome; within-day high-minus-low with Newey-West t; inside the frozen top 20.
  4. Reserved days: v5 scores for the signal days only (no outcomes), for later shadow use.
Descriptive, reused_holdout for the 20-day part (the confirmation window has been read before).
Output: output/experiments/pump_v5_20261007/{eval_report.txt, reserved_scores.parquet}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from experiment_cross_filter_v2 import nw_se  # noqa: E402
from s20_pure_history import outcomes  # noqa: E402
from stockagent_analysis.pump_labels import CleanStart, clean_start_labels  # noqa: E402
from stockagent_analysis.s20_pure import PureConfig, select  # noqa: E402
from train_s20_unified_v1 import scale_free  # noqa: E402

V5 = ROOT / "output/experiments/pump_v5_20261007"
SPLIT = "20260522"
RESERVED_FROM = "20260806"


def price_labels(start: str, end: str) -> pd.DataFrame:
    parts = []
    for f in sorted((ROOT / "output/tushare_cache/daily").glob("*.parquet")):
        if start <= f.stem <= end:
            x = pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "high", "low", "close", "pre_close"])
            parts.append(x[x.ts_code.str.endswith((".SH", ".SZ"))])
    px = pd.concat(parts, ignore_index=True)
    px["trade_date"] = px.trade_date.astype(str)
    lab = clean_start_labels(px, CleanStart())
    px = px.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    g = px.groupby("ts_code", sort=False)
    nxt = g.open.shift(-1)
    hi = pd.concat([g.high.shift(-k) for k in range(1, 6)], axis=1).max(axis=1, skipna=False)
    lo = pd.concat([g.low.shift(-k) for k in range(1, 6)], axis=1).min(axis=1, skipna=False)
    px["a_up"] = np.where(hi.isna(), np.nan, ((hi / nxt - 1 >= 0.10) & (lo / nxt - 1 >= -0.05)).astype(float))
    px["past5"] = g.close.pct_change(5)
    px["past20"] = g.close.pct_change(20)
    lab = lab.merge(px[["ts_code", "trade_date", "a_up", "past5", "past20"]], on=["ts_code", "trade_date"])
    lab["c_up"] = np.where(lab.label.isna(), np.nan, (lab.label == 2).astype(float))
    lab["c_dn"] = np.where(lab.label.isna(), np.nan, (lab.label == 1).astype(float))
    return lab


def daily_auc(frame: pd.DataFrame, y: str, s: str) -> float:
    vals = [roc_auc_score(g[y], g[s]) for _, g in frame.dropna(subset=[y, s]).groupby("trade_date") if g[y].nunique() == 2]
    return float(np.mean(vals))


def top_precision(frame: pd.DataFrame, y: str, s: str, q: float = 0.05) -> float:
    f = frame.dropna(subset=[y, s])
    top = f[f.groupby("trade_date")[s].rank(pct=True, ascending=False) <= q]
    return float(top[y].mean())


def line(x: pd.DataFrame, label: str) -> str:
    return (f"{label:40s} n {len(x):5d} | 5d clean-up {100 * x.c_up.mean():5.1f}% clean-down {100 * x.c_dn.mean():5.1f}% "
            f"| 20d up {100 * x.up.mean():5.1f}% stop-first {100 * x.stop.mean():5.1f}% per trade {x.ret_v1.mean():+5.2f}")


def main() -> int:
    test = pd.read_parquet(V5 / "test_predictions.parquet")
    lab = price_labels("20251201", "20260805")
    v3c = pd.concat([pd.read_parquet(f) for f in sorted((ROOT / "output/pump_history").glob("pump_scores_*.parquet"))])
    v3c = v3c[v3c.usable][["ts_code", "trade_date", "pump_score", "pump_down_score"]]
    t = test.merge(lab, on=["ts_code", "trade_date", "label"], how="left").merge(v3c, on=["ts_code", "trade_date"], how="inner")
    t["seg"] = np.where(t.trade_date <= SPLIT, "early", "late")
    lines = [f"test rows with both models {len(t):,}, days {t.trade_date.nunique()} ({t.trade_date.min()}..{t.trade_date.max()})", "",
             "== 1. label prediction, full market (daily AUC / daily top-5% precision; base rate) =="]
    for seg in ("early", "late", "all"):
        x = t if seg == "all" else t[t.seg == seg]
        for y, base_name in (("c_up", "clean start-up (C)"), ("c_dn", "clean start-down (C)"), ("a_up", "v3c label up (+10% w/o -5%)")):
            v5s = "p_up" if y != "c_dn" else "p_down"
            v3s = "pump_score" if y != "c_dn" else "pump_down_score"
            lines.append(f"[{seg:5s}] {base_name:30s} base {x[y].mean():.3f} | v5 AUC {daily_auc(x, y, v5s):.3f} top5% {top_precision(x, y, v5s):.3f} "
                         f"| v3c AUC {daily_auc(x, y, v3s):.3f} top5% {top_precision(x, y, v3s):.3f}")
    lines += ["", "== 1b. calibration (late segment): predicted probability decile -> observed rate =="]
    late = t[t.seg == "late"].dropna(subset=["c_up", "c_dn"])
    for p_col, y in (("p_up", "c_up"), ("p_down", "c_dn")):
        dec = pd.qcut(late[p_col], 10, labels=False, duplicates="drop")
        cal = late.groupby(dec).agg(pred=(p_col, "mean"), obs=(y, "mean"))
        lines.append(f"v5 {p_col}: " + " ".join(f"{a:.3f}/{b:.3f}" for a, b in zip(cal.pred, cal.obs)))
    lines += ["", "== 2. what v5 P(up) resembles (mean daily rank correlation) =="]
    for col in ("past5", "past20", "pump_score"):
        rho = t.dropna(subset=[col]).groupby("trade_date").apply(lambda g: g.p_up.corr(g[col], method="spearman")).mean()
        lines.append(f"v5 P(up) ~ {col}: {rho:+.3f}")
    rho = t.groupby("trade_date").apply(lambda g: g.p_down.corr(g.pump_down_score, method="spearman")).mean()
    lines.append(f"v5 P(down) ~ v3c P(down): {rho:+.3f}")

    # ---- S20 survivors
    frozen = pd.read_parquet(ROOT / "output/experiments/s20_frozen_rescore_20261005/frozen_scores.parquet")
    frozen = frozen[(frozen.trade_date >= "20260127") & (frozen.trade_date < RESERVED_FROM)]
    surv = select(frozen, PureConfig(top_k=100), "U15D10")
    out = outcomes()
    out = out[out.trade_date < RESERVED_FROM][["ts_code", "trade_date", "ret_v1", "state"]]
    out["up"], out["stop"] = out.state.isin(["pure_up", "dirty_up"]), out.state == "pure_down"
    s = surv.merge(t[["ts_code", "trade_date", "p_up", "p_down", "pump_score", "c_up", "c_dn", "past5", "seg"]], on=["ts_code", "trade_date"]).merge(out)
    lines += ["", f"== 3. S20 survivors: {len(s)} rows, {s.trade_date.nunique()} days; v5 P(up) ~ stage1 within day "
              f"{s.groupby('trade_date').apply(lambda g: g.p_up.corr(g.stage1_probability, method='spearman')).mean():+.3f}, "
              f"~ v3c P(up) {s.groupby('trade_date').apply(lambda g: g.p_up.corr(g.pump_score, method='spearman')).mean():+.3f} =="]
    for col, name in (("p_up", "v5 P(up)"), ("p_down", "v5 P(down)"), ("pump_score", "v3c P(up)")):
        s["t"] = s.groupby("trade_date")[col].transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=False))
        for k, g in s.groupby("t"):
            lines.append(line(g, f"{name} tercile {['low', 'mid', 'high'][int(k)]}"))
        for seg in ("early", "late", "all"):
            x = s if seg == "all" else s[s.seg == seg]
            hi, lo = x[x.t == 2].groupby("trade_date"), x[x.t == 0].groupby("trade_date")
            for m in ("ret_v1", "stop"):
                d = (hi[m].mean() - lo[m].mean()).dropna()
                lines.append(f"    [{seg:5s}] high minus low {m}: {d.mean():+.3f} (t {d.mean() / nw_se(d.to_numpy()):+.2f}, days {len(d)})")
        top = s[s.list_rank <= 20].copy()
        top["h"] = top.groupby("trade_date")[col].rank(pct=True) > 0.5
        for seg in ("early", "late", "all"):
            x = top if seg == "all" else top[top.seg == seg]
            d = (x[x.h].groupby("trade_date").ret_v1.mean() - x[~x.h].groupby("trade_date").ret_v1.mean()).dropna()
            lines.append(f"    [{seg:5s}] inside the frozen top 20, upper minus lower half per trade: {d.mean():+.3f} (t {d.mean() / nw_se(d.to_numpy()):+.2f})")
        lines.append("")

    # ---- reserved days: signal-day scores only
    from run_s20_pure_v1_shadow import SHADOW, load_factors  # noqa: E402
    meta = json.loads((V5 / "feature_meta.json").read_text(encoding="utf-8"))
    features = json.loads((ROOT / "output/production/s20_pure_v1/features.json").read_text(encoding="utf-8"))
    f = load_factors(SHADOW / "factor_groups", features, RESERVED_FROM)
    f[features] = f[features].astype(np.float32)
    scale_free(f, features)
    booster = lgb.Booster(model_file=str(V5 / "classifier.txt"))
    p = booster.predict(f[meta["feature_cols"]], num_iteration=meta["best_iteration"])
    r = f[["ts_code", "trade_date"]].copy()
    r["p_neutral"], r["p_down"], r["p_up"] = p[:, 0], p[:, 1], p[:, 2]
    r.to_parquet(V5 / "reserved_scores.parquet", index=False)
    lines.append(f"== 4. reserved days scored (no outcomes): {r.trade_date.nunique()} days ({r.trade_date.min()}..{r.trade_date.max()}), {len(r):,} rows")
    text = "\n".join(lines)
    (V5 / "eval_report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
