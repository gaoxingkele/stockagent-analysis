#!/usr/bin/env python
"""S20 stage1, unified recipe v1: one recipe for every walk-forward block and for the final model.

Why: the published development scores came from three fold models that early stopping turned into
different things (1-tree and 247-tree residuals), and the frozen model's ranking part was fitted on
two months. None of them is the same selector, so the development results do not describe the
frozen model. This script trains one recipe the same way everywhere:

  * a single LightGBM classifier on the S20 label `positive20` (no anchor/residual split);
  * an expanding window from 2024-01-02, purged so every training label ends before the block;
  * a fixed number of trees, no early stopping;
  * features in price or volume units are divided by the price or volume level, and features
    that depend on where their history starts (obv, ad) or whose definition differs between
    factor stores (adx) are dropped;
  * a refit every quarter; each block is scored only by the model trained before it.

Research only. Models and scores go to output/experiments/s20_unified_v1_20261005. The frozen
contract and output/production are not touched; signal days from 2026-08-06 on are not read.
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
from analyze_s20_pure_r09_funnel import natr14  # noqa: E402
from s20_frames import BASE_STORE, CONFIRM_STORE, LABELS, load_frame  # noqa: E402

OUT = ROOT / "output/experiments/s20_unified_v1_20261005"
DEV_CACHE = ROOT / "output/experiments/s20_pure_20260928/feat_cache_bps5000.parquet"   # 50% row sample, 2024-01-02..2026-01-26
PRICE_UNIT = ["macd", "macd_signal", "macd_hist", "apo", "lr_slope_20", "lr_slope_60"]   # divided by kama_20
VOLUME_UNIT = ["vstd_20", "wvma_20", "adosc", "obv_diff_20"]                            # divided by vma_20
DROPPED = ["kama_20", "vma_20", "obv", "ad", "adx", "lr_angle_20"]
BLOCKS = [("20250303", "20250531"), ("20250601", "20250831"), ("20250901", "20251130"),
          ("20251201", "20260228"), ("20260301", "20260531"), ("20260601", "20260805")]
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=250, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbosity=-1, seed=20261005, num_threads=8)
ROUNDS = 300


def scale_free(frame: pd.DataFrame, features: list[str]) -> list[str]:
    price = frame["kama_20"].where(frame["kama_20"] > 0)
    volume = frame["vma_20"].where(frame["vma_20"] > 0)
    for f in PRICE_UNIT:
        frame[f] = (frame[f] / price).astype(np.float32)
    for f in VOLUME_UNIT:
        frame[f] = (frame[f] / volume).astype(np.float32)
    return [f for f in features if f not in DROPPED]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    features = json.loads((ROOT / "output/production/s20_pure_v1/features.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(20261005)

    # training rows: the sampled development cache, plus half of the confirmation rows for the later blocks
    train = pd.read_parquet(DEV_CACHE, columns=["ts_code", "trade_date", *features])
    train["trade_date"] = train["trade_date"].astype(str)
    train[features] = train[features].astype(np.float32)
    labels = pd.read_parquet(LABELS, columns=["ts_code", "trade_date", "horizon_end_date", "positive20"])
    labels["trade_date"] = labels["trade_date"].astype(str)
    train = train.merge(labels, on=["ts_code", "trade_date"], how="inner")
    conf = load_frame(CONFIRM_STORE, "20260127", "20260805", features)
    conf[features] = conf[features].astype(np.float32)
    half = conf[rng.random(len(conf)) < 0.5]
    train = pd.concat([train, half[["ts_code", "trade_date", *features, "horizon_end_date", "positive20"]]], ignore_index=True)
    train = train[train.positive20 >= 0].reset_index(drop=True)
    train["horizon_end_date"] = train["horizon_end_date"].astype(str)
    model_features = scale_free(train, features)
    print(f"training pool rows {len(train):,} {train.trade_date.min()}..{train.trade_date.max()} features {len(model_features)}", flush=True)

    # rows to score: the full market
    dev = load_frame(BASE_STORE, "20250303", "20260126", features)
    dev[features] = dev[features].astype(np.float32)
    score = pd.concat([dev, conf], ignore_index=True)
    del dev, conf, half
    scale_free(score, features)
    score["unified"] = np.nan
    score["block"] = ""

    lines, gains = [], []
    for start, end in BLOCKS:
        fit = train[train.horizon_end_date < start]
        booster = lgb.train(PARAMS, lgb.Dataset(fit[model_features], label=fit["positive20"]), num_boost_round=ROUNDS)
        booster.save_model(str(OUT / f"unified_v1_{start}.txt"))
        sel = score.trade_date.between(start, end)
        score.loc[sel, "unified"] = booster.predict(score.loc[sel, model_features])
        score.loc[sel, "block"] = start
        test = score[sel]
        daily_auc = np.mean([roc_auc_score(g.positive20, g.unified) for _, g in test.groupby("trade_date") if g.positive20.nunique() == 2])
        top = test[test.groupby("trade_date").unified.rank(ascending=False, method="first") <= 20]
        gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
        gains.append(gain / gain.sum())
        line = (f"block {start}..{end}: fit rows {len(fit):,} ({fit.trade_date.min()}..{fit.trade_date.max()}, base {fit.positive20.mean():.3f}) "
                f"| test days {test.trade_date.nunique()} daily AUC {daily_auc:.3f} | top-20 hit rate {top.positive20.mean():.3f} vs base {test.positive20.mean():.3f} "
                f"| top gain: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in gains[-1].sort_values(ascending=False).head(6).items()))
        print(line, flush=True)
        lines.append(line)

    nat = natr14().rename(columns={"natr14_self": "natr14"})
    nat["trade_date"] = nat["trade_date"].astype(str)
    out = score[["ts_code", "trade_date", "industry", "block", "unified"]].rename(columns={"unified": "stage1_probability"})
    out = out.merge(nat, on=["ts_code", "trade_date"], how="left")
    out.to_parquet(OUT / "predictions.parquet", index=False)
    mean_gain = pd.concat(gains, axis=1).mean(axis=1).sort_values(ascending=False)
    lines.append("mean gain share over the six models: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in mean_gain.head(15).items()))
    lines.append(f"recipe: {json.dumps(PARAMS)} rounds {ROUNDS}; price-unit features / kama_20: {PRICE_UNIT}; "
                 f"volume-unit features / vma_20: {VOLUME_UNIT}; dropped: {DROPPED}")
    (OUT / "train_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(lines[-2], flush=True)
    print(f"wrote predictions.parquet rows {len(out):,}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
