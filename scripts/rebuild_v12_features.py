#!/usr/bin/env python
"""Rebuild the V12 / R20 feature store for a date range from the local caches.

Same computations production uses (update_factor_lab_from_tushare.py and
update_features_to_0605.py), parameterised by date instead of hard-coded:

  0. append the Tushare moneyflow bulk cache to the per-stock moneyflow cache
  1. factor_lab: compute_factors on daily bars since 20240101; columns that cannot
     be derived from bars are filled from each stock's last factor_groups row
     (exactly what production does); written to
     output/factor_lab_3y/factor_groups_extension/group_XXX_ext_rebuild.parquet
     (production's own *_ext_review.parquet files sort after it, so on shared
     dates V12Scorer keeps production's rows)
  2. amount / moneyflow_v1 / mfk / pyramid_v2 / v7_extras: one file per trading
     day named *_ext_{MMDD}.parquet, which is what V12Scorer looks up; existing
     production files are never overwritten, only missing days are added
  Regime files already cover the range locally (no index download needed).

A manifest of every file written goes to output/r20_history/rebuild_manifest.json.

    python scripts/rebuild_v12_features.py --start 20260127 --end 20260928
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import extract_amount_features as _am  # noqa: E402
import extract_mfk_features as _mfk  # noqa: E402
import extract_pyramid_multiwindow as _pyr  # noqa: E402
import extract_v7_extras as _v7  # noqa: E402
from factor_lab import compute_factors  # noqa: E402
from src.stockagent_analysis.moneyflow.features import compute_features as compute_mf_v1  # noqa: E402

DAILY = ROOT / "output/tushare_cache/daily"
MF_BULK = ROOT / "output/tushare_cache/moneyflow"
MF_CACHE = ROOT / "output/moneyflow/cache"
FL_GROUPS = ROOT / "output/factor_lab_3y/factor_groups"
FL_EXT = ROOT / "output/factor_lab_3y/factor_groups_extension"
AUX = {"amount": (ROOT / "output/amount_features", "amount_features"),
       "mf1": (ROOT / "output/moneyflow", "features"),
       "mfk": (ROOT / "output/mfk_features", "features"),
       "pyr": (ROOT / "output/pyramid_v2", "features"),
       "v7": (ROOT / "output/v7_extras", "features")}
HIST_START = "20240101"


def merge_moneyflow(end: str) -> int:
    bulk = pd.concat([pd.read_parquet(f) for f in sorted(MF_BULK.glob("*.parquet")) if f.stem <= end],
                     ignore_index=True)
    bulk["trade_date"] = bulk.trade_date.astype(str)
    n = 0
    for ts, g in bulk.groupby("ts_code"):
        p = MF_CACHE / f"{ts}.parquet"
        if not p.exists():
            continue
        cur = pd.read_parquet(p)
        cur["trade_date"] = cur.trade_date.astype(str)
        new = g[g.trade_date > cur.trade_date.max()].copy()
        if new.empty:
            continue
        for c in cur.columns:
            if c not in new.columns:
                new[c] = pd.NA
        pd.concat([cur, new[cur.columns]], ignore_index=True).sort_values("trade_date").to_parquet(p, index=False)
        n += len(new)
    return n


def load_daily(end: str) -> dict[str, pd.DataFrame]:
    big = pd.concat([pd.read_parquet(f) for f in sorted(DAILY.glob("*.parquet")) if HIST_START <= f.stem <= end],
                    ignore_index=True)
    big["trade_date"] = big.trade_date.astype(str)
    big = big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return {ts: g.reset_index(drop=True) for ts, g in big.groupby("ts_code")}


def rebuild_factor_lab(daily: dict, start: str, end: str, written: list) -> int:
    total = 0
    for i, p in enumerate(sorted(FL_GROUPS.glob("group_*.parquet")), 1):
        df = pd.read_parquet(p)
        df["trade_date"] = df.trade_date.astype(str)
        last = df.sort_values(["ts_code", "trade_date"]).groupby("ts_code").tail(1)
        templates = {r["ts_code"]: r for r in last.to_dict("records")}
        base_cols = list(df.columns)
        rows = []
        for ts in sorted(templates):
            d = daily.get(ts)
            if d is None or len(d) < 60:
                continue
            try:
                f = compute_factors(d)
            except Exception:  # noqa: BLE001  same tolerance as production
                continue
            f["ts_code"] = ts
            new = f[(f.trade_date >= start) & (f.trade_date <= end)].copy()
            if new.empty:
                continue
            for c in base_cols:
                if c not in new.columns:
                    new[c] = templates[ts].get(c, pd.NA)
            rows.append(new[base_cols])
        if rows:
            out = FL_EXT / f"{p.stem}_ext_rebuild.parquet"
            pd.concat(rows, ignore_index=True).to_parquet(out, index=False)
            written.append(str(out.relative_to(ROOT)))
            total += sum(len(r) for r in rows)
        print(f"  factor_lab [{i}] {p.stem}: {sum(len(r) for r in rows)} rows", flush=True)
    return total


def rebuild_aux(daily: dict, start: str, end: str, written: list, skipped: list) -> dict:
    for mod in (_am, _mfk, _pyr, _v7):
        mod.END = end
    parts = {k: [] for k in AUX}
    for i, (ts, d) in enumerate(sorted(daily.items()), 1):
        if len(d) < 60:
            continue
        ohlc = list(zip(d.trade_date, d.open, d.high, d.low, d.close, d.vol))
        try:
            am = _am.compute_amount_features(ohlc, start, end)
            if not am.empty:
                parts["amount"].append(am.assign(ts_code=ts))
        except Exception:  # noqa: BLE001
            pass
        mp = MF_CACHE / f"{ts}.parquet"
        if mp.exists():
            mf = pd.read_parquet(mp)
            mf["trade_date"] = mf.trade_date.astype(str)
            close = list(zip(d.trade_date, d.close))
            for key, fn in (("mf1", lambda: compute_mf_v1(mf)), ("mfk", lambda: _mfk.compute_mfk(mf, close)),
                            ("pyr", lambda: _pyr.compute_pyramid_v2(mf)), ("v7", lambda: _v7.compute_v7(mf, close))):
                try:
                    x = fn()
                    x = x[(x.trade_date.astype(str) >= start) & (x.trade_date.astype(str) <= end)]
                    if not x.empty:
                        parts[key].append(x)
                except Exception:  # noqa: BLE001
                    pass
        if i % 1000 == 0:
            print(f"  aux [{i}/{len(daily)}]", flush=True)
    counts = {}
    for key, (out_dir, name) in AUX.items():
        if not parts[key]:
            continue
        df = pd.concat(parts[key], ignore_index=True)
        df["trade_date"] = df.trade_date.astype(str)
        n = 0
        for day, g in df.groupby("trade_date"):
            out = out_dir / f"{name}_ext_{day[-4:]}.parquet"
            if out.exists():
                skipped.append(str(out.relative_to(ROOT)))
                continue
            g.to_parquet(out, index=False)
            written.append(str(out.relative_to(ROOT)))
            n += 1
        counts[key] = n
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    a = ap.parse_args()
    t0 = time.time()
    written, skipped = [], []
    print(f"0) moneyflow bulk -> per-stock cache: {merge_moneyflow(a.end):,} rows appended", flush=True)
    daily = load_daily(a.end)
    print(f"1) daily: {len(daily)} stocks ({time.time() - t0:.0f}s)", flush=True)
    n = rebuild_factor_lab(daily, a.start, a.end, written)
    print(f"   factor_lab rows {n:,} ({time.time() - t0:.0f}s)", flush=True)
    c = rebuild_aux(daily, a.start, a.end, written, skipped)
    print(f"2) aux day-files written {c}, kept production files {len(skipped)} ({time.time() - t0:.0f}s)", flush=True)
    man = ROOT / "output/r20_history/rebuild_manifest.json"
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text(json.dumps({"start": a.start, "end": a.end, "written": written, "kept_production": skipped},
                              indent=1), encoding="utf-8")
    print(f"manifest -> {man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
