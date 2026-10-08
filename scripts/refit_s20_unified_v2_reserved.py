#!/usr/bin/env python
"""Quarterly refit of unified v2 for the reserved window: one more block, lists only, no outcomes.

Follows train_s20_unified_v2.py to the letter (same training pool, same half of the confirmation
window, same recipe and seed) and adds the block that the shadow rule S0007 needs:

  block 20260806..latest  fit on every purged label whose horizon ends before 2026-08-06
                          (dev cache + the confirmation half), i.e. nothing from the reserved window

The reserved days are then scored and put through the frozen funnel (top 100 -> drop the 40% highest
NATR -> top 20). Only the lists are written. Nothing here reads a reserved-window label or price
path after the signal day; the scoring of S0007 waits for the shadow's maturity.

Output: output/experiments/s20_unified_v2_20261005/reserved/{unified_v2_20260806.txt, predictions.parquet,
daily_lists.csv, refit_report.txt}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_s20_pure_r09_funnel import natr14  # noqa: E402
from s20_frames import CONFIRM_STORE, LABELS, MARKET_FEATURES, load_frame  # noqa: E402
from stockagent_analysis.s20_pure import PureConfig, select  # noqa: E402
from train_s20_unified_v1 import DEV_CACHE, ROUNDS, scale_free  # noqa: E402
from train_s20_unified_v2 import OUT as V2_OUT, PARAMS, day_ranks  # noqa: E402

SHADOW_STORE = ROOT / "output/experiments/s20_pure_v1_shadow/factor_groups"  # rebuilt factors 2026-07-27..latest
BLOCK_START = "20260806"
OUT = V2_OUT / "reserved"


def load_unlabelled(factor_dir: Path, start: str, features: list[str]) -> pd.DataFrame:
    """Signal-day features only: no labels, no regime columns (v2 does not use them), ST names dropped."""
    raw = [f for f in features if f not in MARKET_FEATURES and f != "industry_id"]
    parts = []
    for path in sorted(factor_dir.glob("group_*.parquet")):
        part = pd.read_parquet(path, columns=["ts_code", "trade_date", "industry", *raw])
        part["trade_date"] = part["trade_date"].astype(str)
        part = part[part.trade_date >= start]
        part[raw] = part[raw].apply(pd.to_numeric, errors="coerce").astype(np.float32)
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    frame["ts_code"] = frame["ts_code"].astype(str)
    meta = json.loads((ROOT / "output/lgbm_maxgain/feature_meta.json").read_text(encoding="utf-8"))
    industry_map = meta.get("industry_map", {})
    frame["industry_id"] = frame["industry"].fillna("unknown").astype(str).map(lambda v: industry_map.get(v, -1))
    basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet", columns=["ts_code", "name"])
    st = set(basic.loc[basic["name"].fillna("").str.contains("ST", regex=False), "ts_code"].astype(str))
    frame = frame[~frame.ts_code.isin(st)].copy()
    for f in features:
        if f not in frame:
            frame[f] = np.nan
    return frame.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    features = json.loads((ROOT / "output/production/s20_pure_v1/features.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(20261005)

    # training pool, identical to train_s20_unified_v2.py
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
    del conf, half

    fit = train[train.horizon_end_date < BLOCK_START]
    group = fit.groupby("trade_date", sort=False).size().to_numpy()
    booster = lgb.train(PARAMS, lgb.Dataset(fit[kept], label=fit["positive20"], group=group), num_boost_round=ROUNDS)
    booster.save_model(str(OUT / f"unified_v2_{BLOCK_START}.txt"))
    fit_line = (f"block {BLOCK_START}..: fit rows {len(fit):,} days {len(group)} ({fit.trade_date.min()}..{fit.trade_date.max()}), "
                f"last horizon end {fit.horizon_end_date.max()}; features {len(kept)}")
    print(fit_line, flush=True)
    del train, fit

    # reserved-window signal days: features only
    score = load_unlabelled(SHADOW_STORE, BLOCK_START, features)
    scale_free(score, features)
    day_ranks(score, ranked)
    score["stage1_probability"] = booster.predict(score[kept])
    score["block"] = BLOCK_START
    nat = natr14().rename(columns={"natr14_self": "natr14"})
    nat["trade_date"] = nat["trade_date"].astype(str)
    score = score[["ts_code", "trade_date", "industry", "block", "stage1_probability"]].merge(nat, on=["ts_code", "trade_date"], how="left")
    score.to_parquet(OUT / "predictions.parquet", index=False)

    lists = select(score, PureConfig(), "U15D10")
    basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet", columns=["ts_code", "name"])
    lists = lists.merge(basic, on="ts_code", how="left")
    cols = ["trade_date", "rule", "list_rank", "ts_code", "name", "industry", "stage1_probability", "pool_rank", "natr14", "natr_pct_in_pool"]
    lists[cols].sort_values(["trade_date", "list_rank"]).to_csv(OUT / "daily_lists.csv", index=False, encoding="utf-8-sig")

    frozen = pd.read_csv(ROOT / "output/experiments/s20_pure_v1_shadow/daily_lists.csv", dtype={"trade_date": str})
    frozen = frozen[(frozen.rule == "U15D10") & (frozen.served_by == "U15D10")]
    overlap = []
    for day, g in lists.groupby("trade_date"):
        f = set(frozen[frozen.trade_date == day].ts_code)
        if f:
            overlap.append(len(set(g.ts_code) & f) / 20)
    gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
    gain = (gain / gain.sum()).sort_values(ascending=False)
    lines = [fit_line,
             f"scored signal days {score.trade_date.nunique()} ({score.trade_date.min()}..{score.trade_date.max()}), rows {len(score):,}; lists written for {lists.trade_date.nunique()} days",
             f"name overlap with the frozen list, same day: mean {100 * np.mean(overlap):.1f}% over {len(overlap)} days",
             f"list industry mix: {lists.industry.value_counts(normalize=True).head(8).round(3).to_dict()}",
             "top gain: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in gain.head(10).items()),
             f"recipe: {json.dumps(PARAMS)} rounds {ROUNDS}; market-level features dropped; stock-level features ranked inside the day",
             "no label or post-signal price of the reserved window was read; outcomes are scored only at the shadow's maturity"]
    (OUT / "refit_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[1:5]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
