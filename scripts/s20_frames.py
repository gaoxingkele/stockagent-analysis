#!/usr/bin/env python
"""Full-market S20 feature frames for any date range, built the way the confirmation run builds them.

`confirm_s20_20r_portable._load_confirmation` reads a whole factor directory. This module does the
same joins (labels, market regime, industry id, ST exclusion) but filters by date while reading, so
the development months can be loaded for the full market from the base factor store as well.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE_STORE = ROOT / "output/factor_lab_3y/factor_groups"                       # through 2026-01-26
CONFIRM_STORE = ROOT / "output/experiments/s20_20r_confirmation/factor_groups"  # 2026-01-27 .. 2026-08-05
LABELS = ROOT / "output/experiments/s20_v2_labels/labels.parquet"
MARKET_FEATURES = {
    "regime_id", "mkt_ret_5d", "mkt_ret_20d", "mkt_ret_60d", "mkt_rsi14", "mkt_vol_ratio", "regime_days_in",
    "regime_intensity", "hs300_ret60_z60", "cyb_rel_strength", "zz500_rel_strength",
}
RESERVED_FROM = "20260806"


def load_frame(factor_dir: Path, start: str, end: str, features: list[str]) -> pd.DataFrame:
    if end >= RESERVED_FROM:
        raise PermissionError("the reserved window is not loaded here")
    raw = [f for f in features if f not in MARKET_FEATURES and f != "industry_id"]
    parts = []
    for path in sorted(factor_dir.glob("group_*.parquet")):
        part = pd.read_parquet(path, columns=["ts_code", "trade_date", "industry", *raw])
        part["trade_date"] = part["trade_date"].astype(str)
        part = part[part.trade_date.between(start, end)]
        part[raw] = part[raw].apply(pd.to_numeric, errors="coerce").astype(np.float32)
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    frame["ts_code"] = frame["ts_code"].astype(str)
    labels = pd.read_parquet(LABELS, columns=["ts_code", "trade_date", "horizon_end_date", "positive20", "class20"])
    labels["ts_code"] = labels["ts_code"].astype(str)
    labels["trade_date"] = labels["trade_date"].astype(str)
    frame = frame.merge(labels, on=["ts_code", "trade_date"], how="inner", validate="one_to_one")

    regime = pd.read_parquet(ROOT / "output/regimes/daily_regime.parquet").rename(columns={
        "ret_5d": "mkt_ret_5d", "ret_20d": "mkt_ret_20d", "ret_60d": "mkt_ret_60d", "rsi14": "mkt_rsi14",
        "vol_ratio": "mkt_vol_ratio"})
    regime["trade_date"] = regime["trade_date"].astype(str)
    extra = pd.read_parquet(ROOT / "output/regime_extra/regime_extra.parquet")
    extra["trade_date"] = extra["trade_date"].astype(str)
    regime = regime.merge(extra, on="trade_date", how="left")
    frame = frame.merge(regime[["trade_date", *[f for f in features if f in regime.columns]]],
                        on="trade_date", how="left", validate="many_to_one")
    meta = json.loads((ROOT / "output/lgbm_maxgain/feature_meta.json").read_text(encoding="utf-8"))
    industry_map = meta.get("industry_map", {})
    frame["industry_id"] = frame["industry"].fillna("unknown").astype(str).map(lambda v: industry_map.get(v, -1))
    basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet", columns=["ts_code", "name"])
    st = set(basic.loc[basic["name"].fillna("").str.contains("ST", regex=False), "ts_code"].astype(str))
    frame = frame[~frame.ts_code.isin(st) & (frame.positive20 >= 0)].copy()
    for f in features:
        if f not in frame:
            frame[f] = np.nan
        frame[f] = pd.to_numeric(frame[f], errors="coerce")
    return frame.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
