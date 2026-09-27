#!/usr/bin/env python
"""Train unS20 from scratch on down-first labels. Not a promotion run."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from explore_r20_target_prob_v2 import _load_dataset  # noqa: E402
from stockagent_analysis.r20_target_prob import probability_metrics  # noqa: E402
from stockagent_analysis.s20 import daily_topk_metrics, purged_walk_forward_masks  # noqa: E402
from train_s20_20r_residual import PORTABLE_EXCLUDED_FEATURES  # noqa: E402
from train_s20_v2_multitarget import (  # noqa: E402
    COMPARABLE_SAMPLE_SEED,
    S20_V2_FOLDS,
    _apply_platt,
    _fit_binary,
    _fit_platt,
)

LABELS = ROOT / "output/experiments/s20_uns20_20260927/labels.parquet"
OUT = ROOT / "output/experiments/s20_uns20_20260927"
SEEDS = (20260927, 20261011, 20261025)
TARGET = "down_first20"


def _prepare(sample_bps: int):
    data, features, source = _load_dataset(sample_bps, COMPARABLE_SAMPLE_SEED)
    labels = pd.read_parquet(
        LABELS,
        columns=["ts_code", "trade_date", "horizon_end_date", TARGET, "down_any20", "reason20"],
    )
    labels["ts_code"] = labels["ts_code"].astype(str)
    labels["trade_date"] = labels["trade_date"].astype(str)
    data = data.drop(columns=[c for c in ("horizon_end_date",) if c in data.columns])
    data = data.merge(labels, on=["ts_code", "trade_date"], how="inner", validate="one_to_one")
    data = data[data["trade_date"] <= "20260126"].copy()
    features = [c for c in features if c not in PORTABLE_EXCLUDED_FEATURES]
    return data, features, source


def main() -> int:
    if not LABELS.exists():
        raise FileNotFoundError(f"build labels first: {LABELS}")
    data, features, source = _prepare(5000)
    OUT.mkdir(parents=True, exist_ok=True)
    probability_rows = []
    topk_rows = []
    parts = []
    for fold in S20_V2_FOLDS:
        masks = purged_walk_forward_masks(data["trade_date"], data["horizon_end_date"], fold)
        valid = data[TARGET] >= 0
        fit = data.loc[masks["fit"] & valid]
        tune = data.loc[masks["tune"] & valid]
        calibration = data.loc[masks["calibration"] & valid]
        test = data.loc[masks["test"] & valid]
        print(
            f"{fold.name}: fit={len(fit):,} tune={len(tune):,} "
            f"cal={len(calibration):,} test={len(test):,}",
            flush=True,
        )
        if min(len(fit), len(tune), len(calibration), len(test)) == 0:
            raise ValueError(f"{fold.name} has an empty split")
        models = []
        fold_dir = OUT / fold.name
        fold_dir.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            model = _fit_binary(fit, tune, features, TARGET, seed, 0)
            model.save_model(str(fold_dir / f"uns20_seed{seed}.txt"))
            models.append(model)
        raw_cal = np.mean([m.predict(calibration[features]) for m in models], axis=0)
        raw_test = np.mean([m.predict(test[features]) for m in models], axis=0)
        platt = _fit_platt(raw_cal, calibration[TARGET].to_numpy())
        calibrated = _apply_platt(platt, raw_test)
        scored = test[["ts_code", "trade_date", TARGET, "down_any20"]].copy()
        scored["fold"] = fold.name
        scored["uns20_p"] = calibrated
        parts.append(scored)
        probability_rows.append(
            {"fold": fold.name, **probability_metrics(test[TARGET], calibrated)}
        )
        for k in (10, 20, 50):
            topk_rows.append(
                {
                    "fold": fold.name,
                    "k": k,
                    **daily_topk_metrics(
                        scored, probability_col="uns20_p", target_col=TARGET, k=k
                    ),
                }
            )
    predictions = pd.concat(parts, ignore_index=True)
    predictions.to_parquet(OUT / "predictions.parquet", index=False)
    summary = {
        "model": "uns20 binary LightGBM ensemble, trained from scratch",
        "target": "down_first20: -10% before +20% inside 20 sessions after T+1 open",
        "excluded": "ambiguous same-session dual touch; positive20==0 is not the label",
        "features": "portable factor set used by S20-20R stage 1",
        "sample_bps": 5000,
        "sample_seed": COMPARABLE_SAMPLE_SEED,
        "rows": int(len(data)),
        "resolved_rate": float((data[TARGET] >= 0).mean()),
        "down_first_rate_resolved": float(data.loc[data[TARGET] >= 0, TARGET].mean()),
        "date_min": str(data["trade_date"].min()),
        "date_max": str(data["trade_date"].max()),
        "confirmation_window_not_used": "20260127-20260805",
        "reserved_window_not_used": "20260806+",
        "promotion_eligible": False,
        "source_rows": source.get("rows"),
        "feature_count": len(features),
        "probability": probability_rows,
        "topk_down_precision": topk_rows,
    }
    (OUT / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
