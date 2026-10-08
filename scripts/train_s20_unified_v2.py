#!/usr/bin/env python
"""S20 stage1, unified recipe v2: rank the stocks of each day.

v1 (train_s20_unified_v1.py) spent about 70% of its gain on market-level features, which are the
same for every stock on a day: it mostly learned which days are easy. v2 keeps v1's walk-forward
protocol and changes only what the model is asked to do:

  * objective: LambdaRank with one query per trade date, label `positive20`, so the loss only
    cares about the order of stocks inside a day;
  * features: market-level features are dropped (they cannot order stocks within a day); every
    stock-level feature is replaced by its percentile rank inside the day, after v1's removal of
    price and volume units, so a shift in the level or spread of a feature over time does not
    move a stock across a split;
  * everything else as v1: expanding window from 2024-01-02, purged labels, 300 trees, no early
    stopping, a refit every quarter, each block scored only by the model trained before it.

Research only. Output: output/experiments/s20_unified_v2_20261005. The reserved window is not read.
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
from s20_frames import BASE_STORE, CONFIRM_STORE, LABELS, MARKET_FEATURES, load_frame  # noqa: E402
from train_s20_unified_v1 import BLOCKS, DEV_CACHE, ROUNDS, scale_free  # noqa: E402

OUT = ROOT / "output/experiments/s20_unified_v2_20261005"
PARAMS = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[20], lambdarank_truncation_level=50,
              learning_rate=0.03, num_leaves=31, min_data_in_leaf=250, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbosity=-1, seed=20261005, num_threads=8)


def day_ranks(frame: pd.DataFrame, cols: list[str]) -> None:
    frame[cols] = frame.groupby("trade_date")[cols].rank(pct=True).astype(np.float32)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    features = json.loads((ROOT / "output/production/s20_pure_v1/features.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(20261005)

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
    train = train[train.positive20 >= 0].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    train["horizon_end_date"] = train["horizon_end_date"].astype(str)
    kept = [f for f in scale_free(train, features) if f not in MARKET_FEATURES]
    ranked = [f for f in kept if f != "industry_id"]
    day_ranks(train, ranked)
    print(f"training pool rows {len(train):,} features {len(kept)} (ranked inside the day: {len(ranked)})", flush=True)

    dev = load_frame(BASE_STORE, "20250303", "20260126", features)
    dev[features] = dev[features].astype(np.float32)
    score = pd.concat([dev, conf], ignore_index=True)
    del dev, conf, half
    scale_free(score, features)
    day_ranks(score, ranked)
    score["v2"] = np.nan
    score["block"] = ""

    lines, gains = [], []
    for start, end in BLOCKS:
        fit = train[train.horizon_end_date < start]
        group = fit.groupby("trade_date", sort=False).size().to_numpy()
        booster = lgb.train(PARAMS, lgb.Dataset(fit[kept], label=fit["positive20"], group=group), num_boost_round=ROUNDS)
        booster.save_model(str(OUT / f"unified_v2_{start}.txt"))
        sel = score.trade_date.between(start, end)
        score.loc[sel, "v2"] = booster.predict(score.loc[sel, kept])
        score.loc[sel, "block"] = start
        test = score[sel]
        daily_auc = np.mean([roc_auc_score(g.positive20, g.v2) for _, g in test.groupby("trade_date") if g.positive20.nunique() == 2])
        top = test[test.groupby("trade_date").v2.rank(ascending=False, method="first") <= 20]
        gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
        gains.append(gain / gain.sum())
        line = (f"block {start}..{end}: fit rows {len(fit):,} days {len(group)} ({fit.trade_date.min()}..{fit.trade_date.max()}) "
                f"| test days {test.trade_date.nunique()} daily AUC {daily_auc:.3f} | top-20 hit rate {top.positive20.mean():.3f} vs base {test.positive20.mean():.3f} "
                f"| top gain: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in gains[-1].sort_values(ascending=False).head(6).items()))
        print(line, flush=True)
        lines.append(line)

    nat = natr14().rename(columns={"natr14_self": "natr14"})
    nat["trade_date"] = nat["trade_date"].astype(str)
    out = score[["ts_code", "trade_date", "industry", "block", "v2"]].rename(columns={"v2": "stage1_probability"})
    out = out.merge(nat, on=["ts_code", "trade_date"], how="left")
    out.to_parquet(OUT / "predictions.parquet", index=False)
    mean_gain = pd.concat(gains, axis=1).mean(axis=1).sort_values(ascending=False)
    lines.append("mean gain share over the six models: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in mean_gain.head(15).items()))
    lines.append(f"recipe: {json.dumps(PARAMS)} rounds {ROUNDS}; market-level features dropped; stock-level features ranked inside the day")
    (OUT / "train_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(lines[-2], flush=True)
    print(f"wrote predictions.parquet rows {len(out):,}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
