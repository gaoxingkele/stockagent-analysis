"""Coverage audit of vendor daily limits; not auction fill evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import uuid

import numpy as np
import pandas as pd

from .runtime import atomic_json, digest, load_plan, now


def inspect_limits(frame, date, expected_codes):
    required = {"trade_date", "ts_code", "up_limit", "down_limit"}
    if not required.issubset(frame.columns):
        raise ValueError("missing limit fields")
    up = pd.to_numeric(frame.up_limit, errors="coerce")
    down = pd.to_numeric(frame.down_limit, errors="coerce")
    valid = np.isfinite(up) & np.isfinite(down) & up.gt(0) & down.gt(0) & up.ge(down)
    duplicate = frame.ts_code.duplicated(keep=False)
    correct_date = frame.trade_date.astype(str).eq(date)
    good_codes = set(frame.loc[valid & ~duplicate & correct_date, "ts_code"].astype(str))
    return {"date": date, "rows": len(frame), "invalid_limit_rows": int((~valid).sum()),
            "duplicate_code_rows": int(duplicate.sum()), "wrong_date_rows": int((~correct_date).sum()),
            "expected_codes": len(expected_codes), "usable_expected_codes": len(set(expected_codes) & good_codes),
            "missing_or_invalid_expected_codes": sorted(set(expected_codes) - good_codes),
            "raw_missing_expected_codes": sorted(set(expected_codes) - set(frame.ts_code.astype(str)))}


def audit(root: Path, source: Path):
    root, source = Path(root).resolve(), Path(source).resolve()
    files = sorted(source.glob("*.parquet"))
    plan_path = source / "collection_plan.json"
    plan_hash = digest(plan_path) if plan_path.exists() else None
    plan = load_plan(plan_path) if plan_path.exists() else None
    planned = {}
    if plan is not None:
        if plan.get("api") != "stk_limit":
            raise ValueError("wrong acquisition API")
        planned = {item["date"]: item for item in plan["daily"]}
        if len(planned) != len(plan["daily"]):
            raise ValueError("duplicate planned date")
    by_date = {}
    for path in files:
        match = re.fullmatch(r"(?:stk_limit_)?(\d{8})", path.stem)
        if not match or match[1] in by_date:
            raise ValueError("invalid or duplicate limit date file")
        by_date[match[1]] = path
    daily = sorted((root / "output/tushare_cache/daily").glob("*.parquet"))
    if not daily:
        raise ValueError("no reference daily data")
    if plan is not None and set(planned) != {p.stem for p in daily}:
        raise ValueError("acquisition and reference date inventory differ")
    diagnostics, inputs, missing_dates = [], [], []
    for price_path in daily:
        price_hash = digest(price_path)
        codes = set(pd.read_parquet(price_path, columns=["ts_code"]).ts_code.astype(str))
        if digest(price_path) != price_hash:
            raise ValueError("daily input changed")
        date = price_path.stem
        if plan is not None and planned[date]["sha256"] != price_hash:
            raise ValueError("frozen daily input changed")
        path = by_date.get(date)
        receipt = {"daily_path": str(price_path), "daily_sha256": price_hash}
        if path is None:
            missing_dates.append(date)
            diagnostics.append({"date": date, "status": "missing_file", "expected_codes": len(codes),
                                "usable_expected_codes": 0, "missing_or_invalid_expected_codes": sorted(codes)})
        else:
            source_hash = digest(path)
            saved = None
            if plan is not None:
                receipt_path = source / (date + ".json")
                receipt_hash = digest(receipt_path)
                saved = load_plan(receipt_path)
                if saved.get("date") != date or saved.get("sha256") != source_hash or saved.get("possible_truncation") is not False:
                    raise ValueError("invalid acquisition receipt or data hash")
                requested, received = pd.Timestamp(saved["requested_at"]), pd.Timestamp(saved["received_at"])
                if pd.isna(requested) or pd.isna(received) or requested.tzinfo is None or received.tzinfo is None or received < requested:
                    raise ValueError("invalid receipt timing")
                receipt.update(receipt_sha256=receipt_hash, requested_at=saved["requested_at"], received_at=saved["received_at"])
            frame = pd.read_parquet(path)
            if digest(path) != source_hash:
                raise ValueError("limit input changed")
            receipt.update(limit_path=str(path), limit_sha256=source_hash)
            check = inspect_limits(frame, date, codes)
            if saved is not None:
                if len(frame) >= plan["row_cap"] or check != saved["validation"]:
                    raise ValueError("receipt validation mismatch or truncated data")
                if digest(receipt_path) != receipt_hash:
                    raise ValueError("receipt changed during audit")
            diagnostics.append(dict(check, status="inspected"))
        inputs.append(receipt)
    if plan_hash is not None and digest(plan_path) != plan_hash:
        raise ValueError("collection plan changed during audit")
    output = root / "output/experiments/s20_safe_v4/sources" / ("limit-audit-" + uuid.uuid4().hex)
    output.mkdir(parents=True)
    atomic_json(output / "inputs.json", inputs)
    atomic_json(output / "date_diagnostics.json", diagnostics)
    report = {"at": now(), "directory": str(output), "source": str(source),
              "daily_dates": len(daily), "limit_files": len(files), "missing_dates": missing_dates,
              "extra_dates": sorted(set(by_date) - {p.stem for p in daily}),
              "expected_stock_dates": sum(d["expected_codes"] for d in diagnostics),
              "usable_stock_dates": sum(d["usable_expected_codes"] for d in diagnostics),
              "invalid_limit_rows": sum(d.get("invalid_limit_rows", 0) for d in diagnostics),
              "duplicate_code_rows": sum(d.get("duplicate_code_rows", 0) for d in diagnostics),
              "wrong_date_rows": sum(d.get("wrong_date_rows", 0) for d in diagnostics),
              "acquisition_receipts_verified": plan is not None and not missing_dates,
              "collection_plan_sha256": plan_hash,
              "inputs_sha256": digest(output / "inputs.json"), "code_sha256": digest(Path(__file__)),
              "historical_availability_proven": False, "auction_fill_proven": False,
              "formal_H01_gate_passed": False,
              "unknown_policy": "Absent limits do not imply unbounded trading, no ST or no-limit IPO status."}
    atomic_json(output / "summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    result = audit(Path(__file__).resolve().parents[2], args.source)
    print(json.dumps({k: v for k, v in result.items() if k != "missing_dates"}, indent=2))
    print("missing_date_count", len(result["missing_dates"]))
