#!/usr/bin/env python
"""pump v5b: two separate robust binary heads for the clean start-up and the clean start-down (method C).

Follows from train_pump_v5_shape.py (2026-10-07): the multiclass model put 17.6% of its gain on
industry_id and ranked worse on the test window than single features did. Per quarter, clean start-down
was predictable in all 11 quarters (ma_ratio_120 / rsi_24 daily AUC 0.52..0.68, never flipping) while
clean start-up flipped with the regime (0.47..0.62). So v5b:
  * one binary head per side, scored by daily AUC,
  * no industry_id (it memorises which sectors moved in the training window), no market features,
  * low capacity (15 leaves, 2000 rows per leaf, 300 trees at learning rate 0.03): fewer, broader rules.
Windows, labels and feature preparation are those of train_pump_v5_shape.py. Tree count fixed in advance.

Output: output/experiments/pump_v5_20261007/v5b/{up.txt, down.txt, report.txt, test_predictions.parquet}
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
import train_pump_v5_shape as V5  # noqa: E402
from s20_frames import CONFIRM_STORE, load_frame  # noqa: E402
from train_s20_unified_v1 import DEV_CACHE, scale_free  # noqa: E402

OUT = V5.OUT / "v5b"
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=2000, feature_fraction=0.6,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbosity=-1, seed=20261007, num_threads=8)
ROUNDS = 300


def daily_auc(frame: pd.DataFrame, y: str, s: str) -> float:
    vals = [roc_auc_score(g[y], g[s]) for _, g in frame.groupby("trade_date") if g[y].nunique() == 2]
    return float(np.mean(vals))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    days = V5.trading_days()
    pos = {d: i for i, d in enumerate(days)}
    features = json.loads((ROOT / "output/production/s20_pure_v1/features.json").read_text(encoding="utf-8"))
    lab = V5.labels("20231201", "20260805")
    lab["horizon_end"] = lab.trade_date.map(lambda d: days[pos[d] + 5] if pos[d] + 5 < len(days) else "99999999")

    dev = pd.read_parquet(DEV_CACHE, columns=["ts_code", "trade_date", *features])
    dev["trade_date"] = dev.trade_date.astype(str)
    dev[features] = dev[features].astype(np.float32)
    dev = dev.merge(lab[["ts_code", "trade_date", "label", "horizon_end"]], on=["ts_code", "trade_date"])
    dev = dev[dev.label.notna()]
    feats = [f for f in V5.model_features(features, dev) if f != "industry_id"]
    train = dev[dev.horizon_end <= V5.TRAIN_HORIZON_END]
    valid = dev[(dev.trade_date >= V5.VALID_START) & (dev.horizon_end <= V5.VALID_HORIZON_END)]
    test = load_frame(CONFIRM_STORE, V5.TEST_START, V5.TEST_END, features)
    test[features] = test[features].astype(np.float32)
    scale_free(test, features)
    test = test.merge(lab[["ts_code", "trade_date", "label"]], on=["ts_code", "trade_date"])
    test = test[test.label.notna()]

    lines = [f"features {len(feats)} (no industry_id, no market features); {ROUNDS} trees, 15 leaves, 2000 rows/leaf"]
    preds = test[["ts_code", "trade_date", "label"]].copy()
    for side, cls in (("up", 2), ("down", 1)):
        b = lgb.train(PARAMS, lgb.Dataset(train[feats], label=(train.label == cls).astype(int)), num_boost_round=ROUNDS)
        b.save_model(str(OUT / f"{side}.txt"))
        for name, part in (("valid", valid), ("test", test)):
            x = part[["trade_date"]].assign(y=(part.label == cls).astype(int).to_numpy(), s=b.predict(part[feats]))
            q = x.assign(q=x.trade_date.str[:4] + "Q" + ((x.trade_date.str[4:6].astype(int) - 1) // 3 + 1).astype(str))
            per_q = {k: round(daily_auc(g, "y", "s"), 3) for k, g in q.groupby("q")}
            top = x[x.groupby("trade_date").s.rank(pct=True, ascending=False) <= 0.05]
            lines.append(f"{side:4s} {name}: daily AUC {daily_auc(x, 'y', 's'):.3f} by quarter {per_q} | base {x.y.mean():.3f} top-5% {top.y.mean():.3f}")
            if name == "test":
                preds[f"p_{side}"] = x.s.to_numpy()
        gain = pd.Series(b.feature_importance("gain"), index=b.feature_name())
        gain = (gain / gain.sum()).sort_values(ascending=False)
        lines.append(f"{side:4s} top gain: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in gain.head(10).items()))
    preds.to_parquet(OUT / "test_predictions.parquet", index=False)
    (OUT / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
