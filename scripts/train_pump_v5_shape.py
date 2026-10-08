#!/usr/bin/env python
"""pump v5, shape edition: clean start-up / start-down (method C labels), trained on K-line features only.

Why a local edition: the v3c feature set needs 2023-2025 money-flow, mfk and pyramid tables that only
the production machine has (scripts/../train_pump_classifier_3way_v5.py is the full-feature edition).
Here the features are the S20 technical store: moving-average ratios, oscillators, volume-price and the
61 TA-Lib candlestick patterns, after removing price and volume units (as in the unified S20 recipe),
without market-level features (they cannot tell stocks apart on a day).

Windows (no row is used twice):
  train   signal days 2024-01-02.. whose 5-session horizon ends on or before 2025-09-30  (dev cache, 50% rows)
  valid   signal days 2025-10-01.. whose horizon ends on or before 2026-01-26            (dev cache) tree count only
  test    signal days 2026-01-27..2026-07-28 (horizon ends by 2026-08-05), full market   (confirmation store)
The reserved window is not read here.

Output: output/experiments/pump_v5_20261007/{classifier.txt, feature_meta.json, train_report.txt,
        valid_predictions.parquet, test_predictions.parquet}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from s20_frames import CONFIRM_STORE, MARKET_FEATURES, load_frame  # noqa: E402
from stockagent_analysis.pump_labels import CleanStart, clean_start_labels  # noqa: E402
from train_s20_unified_v1 import DEV_CACHE, scale_free  # noqa: E402

OUT = ROOT / "output/experiments/pump_v5_20261007"
SPEC = CleanStart()
TRAIN_HORIZON_END = "20250930"
VALID_START, VALID_HORIZON_END = "20251001", "20260126"
TEST_START, TEST_END = "20260127", "20260728"
ROUNDS = 200
PARAMS = dict(objective="multiclass", num_class=3, metric="multi_logloss", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=300, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5, lambda_l1=0.1,
              lambda_l2=0.1, max_bin=127, verbosity=-1, seed=20261007, num_threads=8)


def trading_days() -> list[str]:
    return sorted(p.stem for p in (ROOT / "output/tushare_cache/daily").glob("*.parquet"))


def labels(start: str, end: str) -> pd.DataFrame:
    parts = []
    for f in sorted((ROOT / "output/tushare_cache/daily").glob("*.parquet")):
        if start <= f.stem <= end:
            x = pd.read_parquet(f, columns=["ts_code", "trade_date", "high", "low", "close", "pre_close"])
            parts.append(x[x.ts_code.str.endswith((".SH", ".SZ"))])
    px = pd.concat(parts, ignore_index=True)
    px["trade_date"] = px.trade_date.astype(str)
    return clean_start_labels(px, SPEC)


def model_features(features: list[str], frame: pd.DataFrame) -> list[str]:
    kept = scale_free(frame, features)
    return [f for f in kept if f not in MARKET_FEATURES]


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    days = trading_days()
    pos = {d: i for i, d in enumerate(days)}
    horizon_end = lambda d: days[pos[d] + SPEC.horizon] if pos[d] + SPEC.horizon < len(days) else "99999999"  # noqa: E731
    features = json.loads((ROOT / "output/production/s20_pure_v1/features.json").read_text(encoding="utf-8"))

    lab = labels("20231201", "20260805")
    lab["horizon_end"] = lab.trade_date.map(horizon_end)
    print(f"labels: {lab.label.notna().sum():,} rows; class shares {lab.label.value_counts(normalize=True).round(4).to_dict()}", flush=True)

    dev = pd.read_parquet(DEV_CACHE, columns=["ts_code", "trade_date", *features])
    dev["trade_date"] = dev.trade_date.astype(str)
    dev[features] = dev[features].astype(np.float32)
    dev = dev.merge(lab[["ts_code", "trade_date", "label", "horizon_end"]], on=["ts_code", "trade_date"], how="inner")
    dev = dev[dev.label.notna()]
    feats = model_features(features, dev)
    train = dev[dev.horizon_end <= TRAIN_HORIZON_END]
    valid = dev[(dev.trade_date >= VALID_START) & (dev.horizon_end <= VALID_HORIZON_END)]
    del dev

    test = load_frame(CONFIRM_STORE, TEST_START, TEST_END, features)
    test[features] = test[features].astype(np.float32)
    scale_free(test, features)
    test = test.merge(lab[["ts_code", "trade_date", "label"]], on=["ts_code", "trade_date"], how="inner")
    test = test[test.label.notna()]

    lines = [f"spec {SPEC}", f"features {len(feats)} (candlestick {sum(f.startswith('cdl') for f in feats)}); market-level dropped; price/volume units removed"]
    for name, part in (("train", train), ("valid", valid), ("test", test)):
        lines.append(f"{name}: rows {len(part):,} days {part.trade_date.nunique()} ({part.trade_date.min()}..{part.trade_date.max()}) "
                     f"classes {part.label.value_counts(normalize=True).sort_index().round(4).to_dict()}")
    print("\n".join(lines), flush=True)

    dtrain = lgb.Dataset(train[feats], label=train.label.astype(int), categorical_feature=["industry_id"] if "industry_id" in feats else "auto")
    dvalid = lgb.Dataset(valid[feats], label=valid.label.astype(int), reference=dtrain)
    # Tree count fixed at 200, chosen on the validation window by mean daily AUC of P(up) and P(down)
    # (2026-10-07: 10/25/50/100/200/400/600 trees gave up 0.534..0.536, down 0.528..0.553, flat from 200).
    # Early stopping on multi_logloss stopped at 9 trees because the class shares drift between windows.
    booster = lgb.train(PARAMS, dtrain, num_boost_round=ROUNDS, valid_sets=[dvalid], callbacks=[lgb.log_evaluation(100)])
    booster.best_iteration = ROUNDS
    booster.save_model(str(OUT / "classifier.txt"))
    lines.append(f"trees {ROUNDS} (fixed; chosen on validation daily AUC)")

    for name, part in (("valid", valid), ("test", test)):
        p = booster.predict(part[feats], num_iteration=booster.best_iteration)
        out = part[["ts_code", "trade_date", "label"]].copy()
        out["p_neutral"], out["p_down"], out["p_up"] = p[:, 0], p[:, 1], p[:, 2]
        out.to_parquet(OUT / f"{name}_predictions.parquet", index=False)
        for cls, col in ((2, "p_up"), (1, "p_down")):
            base = (out.label == cls).mean()
            top = out[out.groupby("trade_date")[col].rank(pct=True, ascending=False) <= 0.05]
            lines.append(f"{name} class {cls}: base {base:.4f}, daily top-5% precision {(top.label == cls).mean():.4f} (lift {(top.label == cls).mean() / base:.2f}x)")

    gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
    gain = (gain / gain.sum()).sort_values(ascending=False)
    lines.append("top gain: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in gain.head(15).items()))
    lines.append(f"candlestick share of gain {100 * gain[[f for f in gain.index if f.startswith('cdl')]].sum():.1f}%")
    lines.append(f"elapsed {time.time() - t0:.0f}s")
    (OUT / "feature_meta.json").write_text(json.dumps({"feature_cols": feats, "spec": SPEC.__dict__, "params": PARAMS,
                                                       "best_iteration": booster.best_iteration,
                                                       "classes": {"0": "neutral", "1": "clean start-down", "2": "clean start-up"},
                                                       "windows": {"train_horizon_end": TRAIN_HORIZON_END, "valid": [VALID_START, VALID_HORIZON_END],
                                                                   "test": [TEST_START, TEST_END]}}, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "train_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-8:]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
