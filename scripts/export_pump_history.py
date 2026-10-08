#!/usr/bin/env python
"""Export full-market pump (start-up / start-down) scores per signal day. RUN ON THE PRODUCTION MACHINE.

Why: the S20 lists carry no pump score, and this machine lacks the money-flow, mfk, pyramid and
long-return feature groups the pump model needs for 2025. The production machine has them.

What it does, per trading day:
  1. V12Scorer.load_factors_for_date(day) - the same cross-section production scores on
  2. V12Scorer.apply_pump_3way           - r5_pump_3way_lgbm_v3c, unchanged, no refit
  3. keeps every stock (not only V7c / R20 lists), so the S20 funnel can be joined locally:
       ts_code, trade_date, pump_score (P up-start), pump_down_score (P down-start),
       pump_neutral_score, ratio = pump_score / (pump_down_score + 0.01)
     plus `usable` / `stale_groups`: the scorer silently falls back to the LATEST day of a feature file
     when the file has no row for the day asked; such a day is marked unusable (it would carry future data),
     plus data-completeness columns: share of the model's features present for that stock, and,
     per feature group, whether the group is present at all (money flow, mfk, pyramid, long return,
     regime). A score built on missing groups is still written, but flagged.

Fairness: the pump model was trained on 2023-01-01..2025-09-30 and validated to 2026-05-22, so
days before 2025-10-01 are in-sample for it. They are exported for reference and flagged.
Days from 2026-08-06 are the reserved window: only signal-day scores are written here, which is
what shadow rule S0012 needs; nothing about later prices is read.

Days before the factor_lab extension store starts (2026-01-27 on the analysis machine) are loaded
from output/factor_lab_3y/factor_groups instead. On the analysis machine 2026-01-27 onward is complete
and matches the production R20 panel exactly; 2025 needs the production machine's base feature files
(output/{amount_features,moneyflow,mfk_features,pyramid_v2,v7_extras}/features.parquet).

Usage (resumable, one parquet per day):
    python scripts/export_pump_history.py --start 20250303 --end 20991231
Send back the two files written at the end:
    output/pump_history/pump_scores_<first>_<last>.parquet  and  .meta.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "output/pump_history"
MODEL = ROOT / "output/production/r5_pump_3way_lgbm_v3c"
PUMP_TRAIN_END = "20250930"
RESERVED_FROM = "20260806"
GROUPS = {  # feature-name prefixes of the groups this machine is known to miss
    "moneyflow": ("sm_net", "mid_net", "lg_net", "elg_net", "main_net", "main_consec", "elg_ratio", "buy_sell_imb", "dispersion"),
    "mfk": ("mfk_",),
    "pyramid": ("pyr_",),
    "microstructure": ("amount_", "amihud", "kyle", "tail_risk", "range_", "drawdown_recovery", "price_impact", "upward_impact"),
    "event": ("f1_", "f2_"),
    "long_return": ("long_return", "industry_return", "relative_strength", "rs_in_decile", "concept_return"),
    "regime": ("regime_", "mkt_", "hs300_", "cyb_", "zz500_"),
}


def trading_days(start: str, end: str) -> list[str]:
    days = sorted(p.stem for p in (ROOT / "output/tushare_cache/daily").glob("*.parquet"))
    return [d for d in days if start <= d <= end]


def sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper() if path.exists() else None


MAIN_STORE = ROOT / "output/factor_lab_3y/factor_groups"  # 2023-01..2026-01-26; the scorer itself reads only the extension


def stale_groups(scorer, day: str) -> list[str]:
    """Groups whose file has no row for `day`. load_factors_for_date then silently uses the LATEST day
    of that file instead, which for a historical day is future data: such scores are flagged unusable."""
    suffix = day[-4:]
    groups = {
        "amount_features": ["output/amount_features/amount_features.parquet", f"output/amount_features/amount_features_ext_{suffix}.parquet"],
        "moneyflow_v1": ["output/moneyflow/features.parquet", f"output/moneyflow/features_ext_{suffix}.parquet"],
        "mfk": ["output/mfk_features/features.parquet", f"output/mfk_features/features_ext_{suffix}.parquet"],
        "pyramid": ["output/pyramid_v2/features.parquet", f"output/pyramid_v2/features_ext_{suffix}.parquet"],
        "v7_extras": ["output/v7_extras/features.parquet", f"output/v7_extras/features_ext_{suffix}.parquet"],
    }  # same list as V12Scorer.load_factors_for_date (cogalpha is not used by the pump model)
    out = []
    for name, paths in groups.items():
        hit = scorer._slice_target(paths, day)
        if hit is None or hit.empty:
            out.append(name)
    return out


def score_day(scorer, day: str, features: list[str]) -> pd.DataFrame:
    original = scorer.ext_dir
    if not any((pd.read_parquet(p, columns=["trade_date"]).trade_date.astype(str) == day).any() for p in sorted(original.glob("*.parquet"))[:1]):
        scorer.ext_dir = MAIN_STORE          # days before the extension store starts
    try:
        df = scorer.load_factors_for_date(day).copy()
    finally:
        scorer.ext_dir = original
    stale = stale_groups(scorer, day)
    # same call and column order as V12Scorer.apply_pump_3way (0 neutral, 1 down-start, 2 up-start); called
    # directly because apply_pump_3way skips the model unless something loaded it earlier in score_market
    proba = scorer.predict_one(df, "r5_pump_3way")
    if proba.ndim != 2 or proba.shape[1] != 3:
        raise RuntimeError(f"unexpected pump output shape {proba.shape}")
    df["pump_neutral_score"], df["pump_down_score"], df["pump_score"] = proba[:, 0], proba[:, 1], proba[:, 2]
    present = [f for f in features if f in df.columns]
    out = df[["ts_code"]].copy()
    out["trade_date"] = day
    for c in ("pump_score", "pump_down_score", "pump_neutral_score"):
        out[c] = pd.to_numeric(df[c], errors="coerce")
    out["stale_groups"] = ",".join(stale)
    out["usable"] = not stale
    out["ratio"] = out["pump_score"] / (out["pump_down_score"] + 0.01)
    out["feature_share"] = (df[present].notna().sum(axis=1) / len(features)).astype(np.float32) if present else 0.0
    for name, prefixes in GROUPS.items():
        cols = [f for f in present if f.startswith(prefixes)]
        out[f"has_{name}"] = df[cols].notna().any(axis=1) if cols else False
    out["pump_in_sample"] = day <= PUMP_TRAIN_END
    out["reserved_window"] = day >= RESERVED_FROM
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="20250303")
    ap.add_argument("--end", default="20991231")
    a = ap.parse_args()
    from stockagent_analysis.v12_scoring import V12Scorer

    features = json.loads((MODEL / "feature_meta.json").read_text(encoding="utf-8"))["feature_cols"]
    day_dir = OUT / "days"
    day_dir.mkdir(parents=True, exist_ok=True)
    scorer = V12Scorer.get(ROOT)
    days = trading_days(a.start, a.end)
    failed = []
    for i, d in enumerate(days, 1):
        f = day_dir / f"{d}.parquet"
        if f.exists():
            continue
        t0 = time.time()
        try:
            rows = score_day(scorer, d, features)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(days)}] {d} FAILED: {exc}", flush=True)
            failed.append(d)
            continue
        rows.to_parquet(f, index=False)
        gaps = [k for k in GROUPS if not rows[f"has_{k}"].any()]
        stale = rows.stale_groups.iloc[0]
        print(f"[{i}/{len(days)}] {d}: {len(rows)} stocks, feature share median {rows.feature_share.median():.2f}"
              f"{', MISSING groups: ' + ','.join(gaps) if gaps else ''}"
              f"{', STALE (latest-day fallback, unusable): ' + stale if stale else ''} ({time.time() - t0:.0f}s)", flush=True)

    parts = [pd.read_parquet(p) for p in sorted(day_dir.glob("*.parquet")) if a.start <= p.stem <= a.end]
    if not parts:
        print("nothing exported")
        return 1
    allrows = pd.concat(parts, ignore_index=True)
    first, last = allrows.trade_date.min(), allrows.trade_date.max()
    dst = OUT / f"pump_scores_{first}_{last}.parquet"
    allrows.to_parquet(dst, index=False)
    by_day = allrows.groupby("trade_date")
    missing_days = {k: sorted(by_day[f"has_{k}"].any().loc[lambda s: ~s].index.tolist()) for k in GROUPS}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = None
    meta = {"file": dst.name, "days": int(allrows.trade_date.nunique()), "rows": len(allrows), "git_commit": commit,
            "model": MODEL.name, "model_sha256": sha(MODEL / "classifier.txt"),
            "pump_training_window": f"20230101-{PUMP_TRAIN_END}; days up to then are in-sample (pump_in_sample=True)",
            "reserved_from": RESERVED_FROM, "failed_days": failed,
            "unusable_days": sorted(allrows.loc[~allrows.usable, "trade_date"].unique().tolist()),
            "feature_share_median": float(allrows.feature_share.median()),
            "days_missing_group": {k: {"count": len(v), "first": v[:5]} for k, v in missing_days.items()}}
    (OUT / f"pump_scores_{first}_{last}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {dst} and its .meta.json - send both files back")
    if failed:
        print(f"{len(failed)} days failed; rerun to retry them: {failed[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
