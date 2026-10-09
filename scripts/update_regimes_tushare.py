#!/usr/bin/env python
"""Extend output/regimes/daily_regime.parquet and output/regime_extra/regime_extra.parquet to the latest day.

daily_regime: extract_regimes.py (repo root) reads TongDaXin files under D:/tdx with a fixed end date;
this runs that code with the index bars taken from the
Tushare cache (output/tushare_cache/index_daily, refreshed by update_daily_caches.py). regime_extra:
the formulas of update_features_to_0605.py, which wrote the current file. Both are built in a
scratch folder first, checks that every overlapping day reproduces the existing files, and only then
appends the new days. Existing rows are never rewritten. Exit code 2 if the overlap does not match.
"""
from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CODES = {("sh", "000300"): "000300.SH", ("sz", "399006"): "399006.SZ", ("sh", "000905"): "000905.SH"}


def tushare_index(market: str, code: str) -> pd.DataFrame:
    d = pd.read_parquet(ROOT / "output/tushare_cache/index_daily" / f"{CODES[(market, code)]}.parquet")
    return pd.DataFrame({"date": d.trade_date.astype(str), "open": d.open, "high": d.high, "low": d.low,
                         "close": d.close, "volume": d.vol.astype(float)}).sort_values("date").reset_index(drop=True)


def regime_extra(hs: pd.DataFrame) -> pd.DataFrame:
    """Same formulas as update_features_to_0605.py, which produced the current regime_extra file
    (extract_regime_extra.py counts days from 0 and uses a different intensity)."""
    hs = hs.sort_values("trade_date").reset_index(drop=True)
    out = hs[["trade_date"]].copy()
    block = (hs["regime"] != hs["regime"].shift()).cumsum()
    out["regime_days_in"] = out.groupby(block).cumcount() + 1
    out["regime_intensity"] = (hs["ret_20d"].abs() * 100).fillna(0)
    out["hs300_ret60_z60"] = (hs["ret_60d"] - hs["ret_60d"].rolling(60).mean()) / hs["ret_60d"].rolling(60).std()
    out["cyb_rel_strength"] = hs["cyb_ret_20d"] - hs["ret_20d"]
    out["zz500_rel_strength"] = hs["zz500_ret_20d"] - hs["ret_20d"]
    return out.replace([np.inf, -np.inf], np.nan)


def compare(old: pd.DataFrame, new: pd.DataFrame, name: str) -> bool:
    old = old.assign(trade_date=old.trade_date.astype(str))
    new = new.assign(trade_date=new.trade_date.astype(str))
    j = old.merge(new, on="trade_date", suffixes=("_o", "_n"))
    j = j[j.trade_date >= "20240101"]
    bad = {}
    for c in old.columns:
        if c == "trade_date" or f"{c}_n" not in j:
            continue
        a, b = j[f"{c}_o"], j[f"{c}_n"]
        if not (pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)):
            diff = (a.astype(str) != b.astype(str)).mean()
        else:
            diff = (~np.isclose(a.astype(float), b.astype(float), rtol=1e-6, atol=1e-6, equal_nan=True)).mean()
        if diff > 0:
            bad[c] = round(float(diff), 4)
    print(f"{name}: {len(j)} overlapping days since 2024; columns differing: {bad or 'none'}")
    return not bad


def main() -> int:
    scratch = Path(tempfile.mkdtemp())
    reg = importlib.import_module("extract_regimes")
    reg.read_tdx_index = tushare_index
    reg.END = "99999999"
    reg.OUT_DIR = scratch
    reg.main()
    new_reg = pd.read_parquet(scratch / "daily_regime.parquet")
    old_reg = pd.read_parquet(ROOT / "output/regimes/daily_regime.parquet")
    ok = compare(old_reg, new_reg, "daily_regime")

    new_ext = regime_extra(new_reg)
    old_ext = pd.read_parquet(ROOT / "output/regime_extra/regime_extra.parquet")
    ok = compare(old_ext, new_ext, "regime_extra") and ok
    if not ok:
        print("overlap does not reproduce the existing files; nothing written")
        return 2
    for old, new, path in ((old_reg, new_reg, ROOT / "output/regimes/daily_regime.parquet"),
                           (old_ext, new_ext, ROOT / "output/regime_extra/regime_extra.parquet")):
        last = str(old.trade_date.astype(str).max())
        add = new[new.trade_date.astype(str) > last]
        if len(add):
            add = add.astype({c: old[c].dtype for c in old.columns if c in add.columns})
            pd.concat([old, add[old.columns]], ignore_index=True).to_parquet(path, index=False)
        print(f"{path.name}: appended {len(add)} day(s) after {last}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
