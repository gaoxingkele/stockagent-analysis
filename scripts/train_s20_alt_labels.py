#!/usr/bin/env python
"""Unified recipe v1 with a different label (2026-10-11 alternative-design trial, directions 1 and 2).

  python scripts/train_s20_alt_labels.py

Everything is as train_s20_unified_v1.py (features, scale-free transform, expanding window from 2024,
purged labels, quarterly blocks, 300 trees, the same seed and rows), except the label:
  atr   +3*ATR before -2*ATR within 20 sessions (1) vs the reverse (0); neither is left out
  exc   industry excess +10% before -10% (1) vs the reverse (0); neither is left out
Labels from build_s20_alt_paths.py. Each block is scored only by the model trained before it.
Output: output/experiments/s20_alt_designs_20261011/pred_<label>.parquet and train_<label>.txt.
Research only; the frozen contract and output/production are not touched; the reserved window is not read.
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
from s20_frames import BASE_STORE, CONFIRM_STORE, load_frame  # noqa: E402
from train_s20_unified_v1 import BLOCKS, DEV_CACHE, PARAMS, ROUNDS, scale_free  # noqa: E402

OUT = ROOT / "output/experiments/s20_alt_designs_20261011"
LABELS = {"atr": "lab_atr", "exc": "lab_exc"}


def main() -> int:
    features = json.loads((ROOT / "output/production/s20_pure_v1/features.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(20261005)
    paths = pd.read_parquet(OUT / "paths.parquet", columns=["ts_code", "trade_date", "horizon_end_date", *LABELS.values()])

    train = pd.read_parquet(DEV_CACHE, columns=["ts_code", "trade_date", *features])
    train["trade_date"] = train["trade_date"].astype(str)
    train[features] = train[features].astype(np.float32)
    conf = load_frame(CONFIRM_STORE, "20260127", "20260805", features)
    conf[features] = conf[features].astype(np.float32)
    half = conf[rng.random(len(conf)) < 0.5]
    train = pd.concat([train, half[["ts_code", "trade_date", *features]]], ignore_index=True)
    train = train.merge(paths, on=["ts_code", "trade_date"], how="inner")
    model_features = scale_free(train, features)

    dev = load_frame(BASE_STORE, "20250303", "20260126", features)
    dev[features] = dev[features].astype(np.float32)
    score = pd.concat([dev, conf], ignore_index=True)[["ts_code", "trade_date", *features]]
    del dev, conf, half
    scale_free(score, features)
    score = score.merge(paths, on=["ts_code", "trade_date"], how="left")

    for name, col in LABELS.items():
        fit_pool = train[train[col] >= 0]
        print(f"[{name}] training pool rows {len(fit_pool):,} base {fit_pool[col].mean():.3f}", flush=True)
        pred = pd.Series(np.nan, index=score.index)
        lines = []
        for start, end in BLOCKS:
            fit = fit_pool[fit_pool.horizon_end_date < start]
            booster = lgb.train(PARAMS, lgb.Dataset(fit[model_features], label=fit[col]), num_boost_round=ROUNDS)
            booster.save_model(str(OUT / f"model_{name}_{start}.txt"))
            sel = score.trade_date.between(start, end)
            pred[sel] = booster.predict(score.loc[sel, model_features])
            test = score[sel].assign(p=pred[sel])
            lab = test[test[col] >= 0]
            auc = np.mean([roc_auc_score(g[col], g.p) for _, g in lab.groupby("trade_date") if g[col].nunique() == 2])
            gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
            gain = (gain / gain.sum()).sort_values(ascending=False)
            line = (f"block {start}..{end}: fit rows {len(fit):,} | test days {test.trade_date.nunique()} daily AUC {auc:.3f} "
                    f"| top gain: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in gain.head(6).items()))
            print(f"[{name}] {line}", flush=True)
            lines.append(line)
        out = score[["ts_code", "trade_date"]].assign(score=pred.astype(np.float32))
        out.to_parquet(OUT / f"pred_{name}.parquet", index=False)
        (OUT / f"train_{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
