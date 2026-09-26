"""Acquire and validate research-only exchange calendars with real receipts."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import uuid

import pandas as pd

from .runtime import atomic_json, digest, now


def validate_calendar(frame, exchange, start, end):
    required = {"exchange", "cal_date", "is_open", "pretrade_date"}
    errors = []
    if not required.issubset(frame.columns):
        return {"valid": False, "errors": ["missing columns"], "rows": len(frame)}
    dates = pd.to_datetime(frame.cal_date.astype(str), format="%Y%m%d", errors="coerce")
    flags = pd.to_numeric(frame.is_open, errors="coerce")
    if dates.isna().any() or frame.cal_date.astype(str).duplicated().any():
        errors.append("invalid or duplicate calendar dates")
    if not frame.exchange.eq(exchange).all():
        errors.append("exchange mismatch")
    if not flags.isin([0, 1]).all():
        errors.append("invalid open flags")
    expected = set(pd.date_range(start, end).strftime("%Y%m%d"))
    actual = set(dates.dropna().dt.strftime("%Y%m%d"))
    if expected != actual:
        errors.append("calendar day coverage mismatch")
    if dates.isna().any() or not flags.isin([0, 1]).all():
        return {"valid": False, "errors": errors, "rows": len(frame),
                "missing_dates": sorted(expected - actual), "extra_dates": sorted(actual - expected)}
    ordered = frame.assign(_date=dates, _flag=flags).sort_values("_date")
    last_open = None
    for row in ordered.itertuples(index=False):
        if last_open is not None and str(row.pretrade_date) != last_open:
            errors.append("pretrade chain mismatch")
            break
        if int(row.is_open) == 1:
            last_open = str(row.cal_date)
    return {"valid": not errors, "errors": errors, "rows": len(frame),
            "open_dates": int(flags.eq(1).sum()), "missing_dates": sorted(expected - actual),
            "extra_dates": sorted(actual - expected)}


def acquire(root, query, start="20240101", end="20260911"):
    """Query is injected for deterministic offline tests; no silent fallback."""
    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    if start_date > end_date or end_date.year - start_date.year > 3:
        raise ValueError("invalid or excessive calendar interval")
    base = root / "output/experiments/s20_safe_v4/sources" / ("calendar-" + uuid.uuid4().hex)
    if not base.resolve().is_relative_to(root.resolve()):
        raise ValueError("output path escapes research root")
    base.mkdir(parents=True)
    receipts, merged, checks = [], [], {}
    for exchange in ("SSE", "SZSE"):
        parts = []
        for year in range(start_date.year, end_date.year + 1):
            lo = max(start_date, pd.Timestamp(year, 1, 1)).strftime("%Y%m%d")
            hi = min(end_date, pd.Timestamp(year, 12, 31)).strftime("%Y%m%d")
            params = {"exchange": exchange, "start_date": lo, "end_date": hi}
            requested_at = now()
            try:
                frame = query(**params)
            except Exception as exc:
                # Provider errors can contain request credentials; only persist type.
                atomic_json(base / "failure.json", {"error_type": type(exc).__name__, "params": params,
                                                      "at": now(), "completed_receipts": receipts})
                raise RuntimeError("calendar provider request failed; sanitized failure receipt saved") from None
            path = base / f"{exchange}_{year}.parquet"
            frame.to_parquet(path, index=False)
            check = validate_calendar(frame, exchange, lo, hi)
            receipts.append({"provider": "Tushare", "api": "trade_cal", "params": params,
                             "requested_at": requested_at, "received_at": now(), "file": path.name,
                             "sha256": digest(path), "validation": check,
                             "historical_publication_timestamp_proven": False})
            atomic_json(base / "receipts.json", receipts)
            if not check["valid"]:
                raise ValueError("calendar schema/coverage invalid: " + str(check["errors"]))
            parts.append(frame)
        frame = pd.concat(parts, ignore_index=True)
        checks[exchange] = validate_calendar(frame, exchange, start, end)
        merged.append(frame)
    calendar = pd.concat(merged, ignore_index=True).sort_values(["exchange", "cal_date"])
    calendar.to_parquet(base / "trade_cal.parquet", index=False)
    daily_dates = {p.stem for p in (root / "output/tushare_cache/daily").glob("*.parquet") if start <= p.stem <= end}
    comparisons = {}
    for exchange in ("SSE", "SZSE"):
        open_dates = set(calendar.loc[calendar.exchange.eq(exchange) & calendar.is_open.astype(int).eq(1), "cal_date"].astype(str))
        comparisons[exchange] = {"calendar_open_missing_daily_file": sorted(open_dates - daily_dates),
                                 "daily_file_on_calendar_closed": sorted(daily_dates - open_dates)}
    result = {"at": now(), "directory": str(base), "calendar_validation": checks,
              "daily_file_comparison": comparisons, "calendar_hash": digest(base / "trade_cal.parquet"),
              "BSE": "not requested; provider documentation does not establish support here",
              "formal_H01_gate_passed": False, "production_cache_changed": False}
    atomic_json(base / "calendar_audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20240101")
    parser.add_argument("--end", default="20260911")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    from dotenv import load_dotenv
    import requests
    load_dotenv(root / ".env", override=False)
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise SystemExit("TUSHARE_TOKEN unavailable; no request made")

    def query(**params):
        response = requests.post("https://api.tushare.pro", json={"api_name": "trade_cal", "token": token,
                                 "params": params, "fields": "exchange,cal_date,is_open,pretrade_date"}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError("provider rejected request")
        return pd.DataFrame(payload["data"]["items"], columns=payload["data"]["fields"])

    import json
    print(json.dumps(acquire(root, query, args.start, args.end), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
