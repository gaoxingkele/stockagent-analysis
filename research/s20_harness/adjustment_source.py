"""Resumable, hash-verified adjustment acquisition. No production cache writes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np
import pandas as pd
import psutil

from .runtime import atomic_json, digest, load_plan, now


def validate_factors(frame, date, expected_codes):
    errors = []
    if not {"ts_code", "trade_date", "adj_factor"}.issubset(frame.columns):
        return {"valid": False, "errors": ["missing columns"], "missing_daily_codes": sorted(expected_codes)}
    factor = pd.to_numeric(frame.adj_factor, errors="coerce")
    if not frame.trade_date.astype(str).eq(date).all():
        errors.append("wrong date")
    if frame.ts_code.duplicated().any():
        errors.append("duplicate code")
    if not (np.isfinite(factor) & factor.gt(0)).all():
        errors.append("invalid factor")
    if frame.empty:
        errors.append("empty response")
    return {"valid": not errors, "errors": errors, "rows": len(frame),
            "missing_daily_codes": sorted(set(expected_codes) - set(frame.ts_code.astype(str))),
            "extra_nontrading_codes": len(set(frame.ts_code.astype(str)) - set(expected_codes))}


def collect(root, query, source_id=None, max_requests=700, wall_seconds=1200, pause=.6):
    if not 1 <= max_requests <= 700 or not 0 < wall_seconds <= 1800 or not 0 <= pause <= 10:
        raise ValueError("invalid collection budget")
    source_id = source_id or "adjustments-" + uuid.uuid4().hex
    if Path(source_id).name != source_id or not source_id.startswith("adjustments-"):
        raise ValueError("invalid source id")
    base = root / "output/experiments/s20_safe_v4/sources" / source_id
    if not base.resolve().is_relative_to(root.resolve()):
        raise ValueError("output escapes repository")
    base.mkdir(parents=True, exist_ok=True)
    lock = base / "active.json"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "process_created": psutil.Process().create_time(), "at": now()}, stream)
    except FileExistsError:
        raise RuntimeError("source has a live or unreconciled owner; inspect process identity before recovery") from None
    start = time.monotonic()
    requests_count = 0
    try:
        plan_path = base / "collection_plan.json"
        if plan_path.exists():
            plan = load_plan(plan_path)
            if plan["collector_hash"] != digest(Path(__file__)):
                raise ValueError("collector changed; use a new source version")
        else:
            daily = sorted((root / "output/tushare_cache/daily").glob("*.parquet"))
            if not daily:
                raise ValueError("no daily inputs")
            plan = {"created_at": now(), "source_id": source_id, "api": "adj_factor",
                    "collector_hash": digest(Path(__file__)), "daily": [
                        {"date": p.stem, "path": p.relative_to(root).as_posix(), "sha256": digest(p)} for p in daily],
                    "request_budget_per_invocation": max_requests, "wall_budget_seconds": wall_seconds,
                    "performance_claim": False, "availability": "retrieved now; historical revision timing unproved"}
            atomic_json(plan_path, plan)
        records = []
        for item in plan["daily"]:
            date = item["date"]
            receipt = base / f"{date}.json"
            output = base / f"{date}.parquet"
            daily_path = root / item["path"]
            if digest(daily_path) != item["sha256"]:
                raise ValueError("daily source changed: " + date)
            if receipt.exists():
                saved = load_plan(receipt)
                if not output.exists() or digest(output) != saved["sha256"]:
                    raise ValueError("cached factor hash mismatch: " + date)
                records.append(saved)
                continue
            if output.exists():
                raise ValueError("uncommitted factor file; inspect instead of overwriting: " + date)
            if requests_count >= max_requests or time.monotonic() - start >= wall_seconds:
                break
            requested_at = now()
            requests_count += 1
            try:
                frame = query(trade_date=date)
            except Exception as exc:
                failure = {"at": now(), "date": date, "error_type": type(exc).__name__,
                           "requests_in_invocation": requests_count}
                atomic_json(base / ("failure-" + uuid.uuid4().hex + ".json"), failure)
                raise RuntimeError("provider failed; sanitized receipt saved; no automatic parameter retry") from None
            codes = pd.read_parquet(daily_path, columns=["ts_code"]).ts_code.astype(str)
            check = validate_factors(frame, date, codes)
            frame.to_parquet(output, index=False)
            saved = {"date": date, "requested_at": requested_at, "received_at": now(),
                     "file": output.name, "sha256": digest(output), "validation": check}
            atomic_json(receipt, saved)
            records.append(saved)
            if not check["valid"]:
                raise ValueError("invalid factor schema; evidence retained")
            if requests_count % 25 == 0 or requests_count == 1:
                print(json.dumps({"source_id": source_id, "completed_dates": len(records),
                                  "total_dates": len(plan["daily"]), "last_date": date}), flush=True)
            atomic_json(base / "heartbeat.json", {"at": now(), "pid": os.getpid(),
                        "process_created": psutil.Process().create_time(), "completed_dates": len(records)})
            if pause:
                time.sleep(pause)
        complete = len(records) == len(plan["daily"])
        summary = {"source_id": source_id, "at": now(), "complete_acquisition": complete,
                   "completed_dates": len(records), "total_dates": len(plan["daily"]),
                   "new_requests": requests_count, "elapsed_seconds": time.monotonic() - start,
                   "missing_daily_code_rows": sum(len(r["validation"]["missing_daily_codes"]) for r in records),
                   "all_schema_valid": all(r["validation"]["valid"] for r in records),
                   "economic_cash_ledger_proven": False, "formal_H01_gate_passed": False}
        atomic_json(base / "collection_summary.json", summary)
        return summary
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-id")
    parser.add_argument("--max-requests", type=int, default=700)
    parser.add_argument("--wall-seconds", type=float, default=1200)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    from dotenv import load_dotenv
    import requests
    load_dotenv(root / ".env", override=False)
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise SystemExit("TUSHARE_TOKEN unavailable")
    session = requests.Session()

    def query(**params):
        response = session.post("https://api.tushare.pro", json={"api_name": "adj_factor", "token": token,
                                "params": params, "fields": "ts_code,trade_date,adj_factor"}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError("provider rejected request")
        return pd.DataFrame(payload["data"]["items"], columns=payload["data"]["fields"])

    print(json.dumps(collect(root, query, args.source_id, args.max_requests, args.wall_seconds), indent=2))


if __name__ == "__main__":
    main()
