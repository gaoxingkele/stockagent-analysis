#!/usr/bin/env python
"""Compare stage1 with and without the volume family on the dev walk-forward folds.

Same 50% sample, same folds, same funnel. Reports per fold:
  - raw ranking quality: daily Top20 precision of positive20 (the stage1 training target)
  - S20 v1 funnel on each score (pool Top100 within the sample -> natr cap 40% -> Top20 of 100,
    scaled to the sample: pool 50, list 10) with band-exit per trade, bad rate and the
    style-matched excess (size x EP x turnover terciles).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from style_audit import characteristics  # noqa: E402

EXP = ROOT / "output/experiments"
OUT = EXP / "ablation_volume"


def main() -> int:
    cols = ["ts_code", "trade_date", "fold", "positive20", "s20_20r_platt"]
    base = pd.read_parquet(EXP / "s20_20r_residual_portable/predictions.parquet", columns=cols)
    abl = pd.read_parquet(OUT / "stage1_no_volume/predictions.parquet", columns=cols)
    for x in (base, abl):
        x["trade_date"] = x.trade_date.astype(str)
    m = base.merge(abl[["ts_code", "trade_date", "s20_20r_platt"]], on=["ts_code", "trade_date"],
                   suffixes=("_base", "_novol"))
    band = pd.read_parquet(EXP / "s20_pure_20260928/band_panel.parquet",
                           columns=["ts_code", "trade_date", "cls_a5_d10", "ret_a5_b15_d10"])
    m = m.merge(band, on=["ts_code", "trade_date"])
    C = characteristics(set(m.trade_date))
    m = m.merge(C[["ts_code", "trade_date", "natr14", "size3", "ep3", "turn3", "turn_pct", "amount_pct"]],
                on=["ts_code", "trade_date"], how="left")
    m["ret"] = m.ret_a5_b15_d10 - 0.3
    m["excess"] = m.ret - m.groupby(["trade_date", "size3", "ep3", "turn3"]).ret.transform("mean")
    corr = m.groupby("trade_date").apply(lambda g: g.s20_20r_platt_base.corr(g.s20_20r_platt_novol, method="spearman")).mean()

    rows = []
    for fold, g in [*m.groupby("fold"), ("all", m)]:
        for name in ("base", "novol"):
            s = f"s20_20r_platt_{name}"
            r = g.groupby("trade_date")[s].rank(ascending=False, method="first")
            prec = g[r <= 20].positive20.mean()
            f = g[r <= 50].copy()                                   # pool 100 of the full market ~ 50 of the 50% sample
            f["np"] = f.groupby("trade_date").natr14.rank(pct=True)
            f = f[f.np.isna() | (f.np <= 0.6)]
            f["lr"] = f.groupby("trade_date")[s].rank(ascending=False, method="first")
            L = f[f.lr <= 10]
            day = L.groupby("trade_date")
            rows.append({"fold": fold, "stage1": name, "top20_precision_pos20%": round(100 * prec, 1),
                         "v1_per_trade%": round(float(day.ret.mean().mean()), 2),
                         "v1_bad%": round(100 * (L.cls_a5_d10 == 2).mean(), 1),
                         "v1_style_matched_pp": round(float(day.excess.mean().mean()), 2),
                         "v1_turn_pct": round(float(L.turn_pct.mean()), 2),
                         "v1_amount_pct": round(float(L.amount_pct.mean()), 2)})
    T = pd.DataFrame(rows)
    text = (f"rows {len(m):,}, days {m.trade_date.nunique()}; mean daily Spearman(base, no-volume) = {corr:.3f}\n\n"
            + T.to_string(index=False))
    (OUT / "stage1_ablation_eval.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
