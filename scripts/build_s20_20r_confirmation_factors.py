#!/usr/bin/env python
"""Build portable factor rows for the sealed S20-20R confirmation window."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import talib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from factor_lab import compute_factors  # noqa: E402
from train_s20_20r_residual import PORTABLE_EXCLUDED_FEATURES  # noqa: E402


REGIME_FEATURES = {
    "regime_id",
    "mkt_ret_5d",
    "mkt_ret_20d",
    "mkt_ret_60d",
    "mkt_rsi14",
    "mkt_vol_ratio",
    "regime_days_in",
    "regime_intensity",
    "hs300_ret60_z60",
    "cyb_rel_strength",
    "zz500_rel_strength",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-start", default="20240102")
    parser.add_argument("--confirmation-start", default="20260127")
    parser.add_argument("--confirmation-end", default="20260805")
    parser.add_argument(
        "--daily-dir", type=Path, default=ROOT / "output/tushare_cache/daily"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=ROOT / "output/factor_lab_3y/factor_groups",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output/experiments/s20_20r_confirmation/factor_groups",
    )
    return parser.parse_args()


def _load_daily(args):
    paths = [
        path
        for path in sorted(args.daily_dir.glob("*.parquet"))
        if args.history_start <= path.stem <= args.confirmation_end
    ]
    if not paths:
        raise FileNotFoundError("no daily cache files cover the requested period")
    columns = [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
    ]
    frame = pd.concat(
        [pd.read_parquet(path, columns=columns) for path in paths],
        ignore_index=True,
    )
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["trade_date"] = frame["trade_date"].astype(str)
    return frame.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    started = time.time()
    meta = json.loads(
        (ROOT / "output/lgbm_maxgain/feature_meta.json").read_text(encoding="utf-8")
    )
    portable = [
        feature
        for feature in meta["feature_cols"]
        if feature not in PORTABLE_EXCLUDED_FEATURES and feature != "industry_id"
    ]
    raw_features = [feature for feature in portable if feature not in REGIME_FEATURES]
    daily = _load_daily(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    group_audits = []
    all_codes = set(daily["ts_code"])
    for group_number, base_path in enumerate(
        sorted(args.base_dir.glob("group_*.parquet")), 1
    ):
        base = pd.read_parquet(base_path, columns=["ts_code", "trade_date", "industry"])
        base["ts_code"] = base["ts_code"].astype(str)
        latest = (
            base.sort_values(["ts_code", "trade_date"])
            .groupby("ts_code", sort=False)
            .tail(1)
        )
        industry = latest.set_index("ts_code")["industry"].to_dict()
        codes = [code for code in latest["ts_code"] if code in all_codes]
        subset = daily[daily["ts_code"].isin(codes)]
        rows = []
        failures = []
        for code, stock in subset.groupby("ts_code", sort=False):
            try:
                factors = compute_factors(stock)
                factors["adx"] = talib.ADX(
                    stock["high"].to_numpy(dtype=float),
                    stock["low"].to_numpy(dtype=float),
                    stock["close"].to_numpy(dtype=float),
                    timeperiod=14,
                )
                factors["ts_code"] = code
                factors["industry"] = industry.get(code, "")
                factors = factors[
                    factors["trade_date"].astype(str).between(
                        args.confirmation_start, args.confirmation_end
                    )
                ].copy()
                for feature in raw_features:
                    if feature not in factors:
                        factors[feature] = np.nan
                rows.append(
                    factors[["ts_code", "trade_date", "industry", *raw_features]]
                )
            except Exception as error:
                failures.append({"ts_code": code, "error": str(error)[:160]})
        output = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        output_path = args.output_dir / base_path.name
        output.to_parquet(output_path, index=False)
        audit = {
            "group": base_path.stem,
            "rows": len(output),
            "symbols": int(output["ts_code"].nunique()) if len(output) else 0,
            "date_min": str(output["trade_date"].min()) if len(output) else None,
            "date_max": str(output["trade_date"].max()) if len(output) else None,
            "failures": failures,
        }
        group_audits.append(audit)
        print(
            f"{group_number:02d}: {base_path.stem} rows={len(output):,} "
            f"failures={len(failures)} elapsed={time.time()-started:.1f}s",
            flush=True,
        )
    report = {
        "contract_version": "s20-20r-portable-confirmation-factors-v1",
        "history_start": args.history_start,
        "confirmation_window": [args.confirmation_start, args.confirmation_end],
        "feature_count_without_regime_or_industry": len(raw_features),
        "rows": sum(row["rows"] for row in group_audits),
        "symbols": len(set().union(*[
            set(pd.read_parquet(args.output_dir / f"group_{i:03d}.parquet", columns=["ts_code"])["ts_code"])
            for i in range(1, len(group_audits) + 1)
            if (args.output_dir / f"group_{i:03d}.parquet").exists()
        ])),
        "groups": group_audits,
    }
    (args.output_dir.parent / "factor_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in report if key != "groups"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
