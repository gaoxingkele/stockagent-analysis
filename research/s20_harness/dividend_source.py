"""Research-only corporate-action receipts, including zero-event dates."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import uuid

import pandas as pd
import psutil

from .runtime import atomic_json, digest, load_plan, now

FIELDS = "ts_code,end_date,ann_date,div_proc,stk_div,stk_bo_rate,stk_co_rate,cash_div,cash_div_tax,record_date,ex_date,pay_date,div_listdate,imp_ann_date"


def validate_events(frame, date):
    missing = set(FIELDS.split(",")) - set(frame.columns)
    errors = []
    if missing:
        errors.append("missing columns: " + ",".join(sorted(missing)))
    if len(frame) >= 2000:
        errors.append("possible provider row cap; complete coverage unproved")
    if not missing and not frame.ex_date.astype(str).eq(date).all():
        errors.append("wrong ex date")
    # Missing dates/rates are retained for economic audit, not filled with zeros.
    missing_cells = frame.isna().sum().to_dict()
    return {"valid": not errors, "errors": errors, "rows": len(frame),
            "missing_field_counts": {str(k): int(v) for k, v in missing_cells.items() if v},
            "economic_ledger_validated": False}


def collect(root, query, source_id=None, max_requests=700, wall_seconds=1200, pause=.7):
    if not 1 <= max_requests <= 700 or not 0 < wall_seconds <= 1800 or not 0 <= pause <= 10:
        raise ValueError("invalid acquisition budget")
    source_id = source_id or "dividends-" + uuid.uuid4().hex
    if Path(source_id).name != source_id or not source_id.startswith("dividends-"):
        raise ValueError("invalid source id")
    base = root / "output/experiments/s20_safe_v4/sources" / source_id
    if not base.resolve().is_relative_to(root.resolve()):
        raise ValueError("unsafe output")
    base.mkdir(parents=True, exist_ok=True)
    lock = base / "active.json"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "process_created": psutil.Process().create_time(), "at": now()}, stream)
    started = time.monotonic()
    count = 0
    records = []
    try:
        plan_path = base / "collection_plan.json"
        if plan_path.exists():
            plan = load_plan(plan_path)
            if plan["collector_hash"] != digest(Path(__file__)):
                raise ValueError("collector changed; new source version required")
        else:
            dates = sorted(p.stem for p in (root / "output/tushare_cache/daily").glob("*.parquet"))
            if not dates:
                raise ValueError("daily dates missing")
            plan = {"api": "dividend", "params_key": "ex_date", "dates": dates, "fields": FIELDS,
                    "created_at": now(), "collector_hash": digest(Path(__file__)),
                    "coverage_scope": "ex dates equal observed market dates; not all company actions or future event announcements",
                    "budget": {"requests": max_requests, "wall_seconds": wall_seconds}}
            atomic_json(plan_path, plan)
        for date in plan["dates"]:
            path = base / f"{date}.parquet"
            receipt_path = base / f"{date}.json"
            if receipt_path.exists():
                receipt = load_plan(receipt_path)
                if not path.exists() or digest(path) != receipt["sha256"]:
                    raise ValueError("cached event hash mismatch")
                records.append(receipt)
                continue
            if path.exists():
                raise ValueError("uncommitted file; do not overwrite")
            if count >= max_requests or time.monotonic() - started >= wall_seconds:
                break
            requested = now()
            count += 1
            try:
                frame = query(ex_date=date)
            except Exception as exc:
                atomic_json(base / ("failure-" + uuid.uuid4().hex + ".json"),
                            {"at": now(), "date": date, "error_type": type(exc).__name__})
                raise RuntimeError("provider failure; sanitized receipt retained") from None
            check = validate_events(frame, date)
            frame.to_parquet(path, index=False)
            receipt = {"date": date, "requested_at": requested, "received_at": now(),
                       "sha256": digest(path), "validation": check}
            atomic_json(receipt_path, receipt)
            records.append(receipt)
            if not check["valid"]:
                raise ValueError("event response validity failed; see receipt")
            atomic_json(base / "heartbeat.json", {"at": now(), "pid": os.getpid(),
                        "process_created": psutil.Process().create_time(), "completed_dates": len(records)})
            if count == 1 or count % 25 == 0:
                print(json.dumps({"source_id": source_id, "completed_dates": len(records), "total_dates": len(plan["dates"])}), flush=True)
            if pause:
                time.sleep(pause)
        result = {"source_id": source_id, "at": now(), "complete_acquisition": len(records) == len(plan["dates"]),
                  "completed_dates": len(records), "total_dates": len(plan["dates"]), "new_requests": count,
                  "event_rows": sum(r["validation"]["rows"] for r in records),
                  "all_schema_valid": all(r["validation"]["valid"] for r in records),
                  "zero_event_dates": sum(r["validation"]["rows"] == 0 for r in records),
                  "formal_H01_gate_passed": False, "economic_ledger_validated": False}
        atomic_json(base / "collection_summary.json", result)
        return result
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
        response = session.post("https://api.tushare.pro", json={"api_name": "dividend", "token": token,
                                "params": params, "fields": FIELDS}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError("provider rejected request")
        return pd.DataFrame(payload["data"]["items"], columns=payload["data"]["fields"])
    print(json.dumps(collect(root, query, args.source_id, args.max_requests, args.wall_seconds), indent=2))


if __name__ == "__main__":
    main()
