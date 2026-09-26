"""Resumable full-calendar suspend_d acquisition, no implicit historical truth."""
import argparse
import json
import os
from pathlib import Path
import re
import time
import uuid

import pandas as pd
import psutil

from .runtime import atomic_json, digest, load_plan, now

FIELDS = "ts_code,trade_date,suspend_timing,suspend_type"


def inspect(frame, date):
    if not set(FIELDS.split(",")).issubset(frame):
        raise ValueError("missing suspension fields")
    valid = (frame.trade_date.astype(str).eq(date) &
             frame.ts_code.astype(str).str.fullmatch(r"\d{6}\.(SH|SZ|BJ)") &
             frame.suspend_type.isin(["S", "R"]))
    return {"rows": len(frame), "invalid_rows": int((~valid).sum()),
            "duplicate_rows": int(frame.duplicated().sum()),
            "type_counts": frame.suspend_type.value_counts().to_dict(),
            "timing_nonnull": int(frame.suspend_timing.notna().sum()),
            "possible_truncation": len(frame) >= 5000}


def collect(root, query, source_id=None, max_requests=700, pause=.6):
    root = Path(root).resolve()
    if not 1 <= max_requests <= 700 or not 0 <= pause <= 10:
        raise ValueError("invalid budget")
    source_id = source_id or "suspension-daily-" + uuid.uuid4().hex
    if not re.fullmatch(r"suspension-daily-[a-f0-9]{32}", source_id):
        raise ValueError("invalid source id")
    output = root / "output/experiments/s20_safe_v4/sources" / source_id
    output.mkdir(parents=True, exist_ok=True)
    identity = {"pid": os.getpid(), "process_created": psutil.Process().create_time()}
    lock = output / "active.json"
    with lock.open("x") as stream:
        json.dump(identity, stream)
    records, used = [], 0
    try:
        plan_path = output / "collection_plan.json"
        if plan_path.exists():
            plan = load_plan(plan_path)
            if plan["code_sha256"] != digest(Path(__file__)):
                raise ValueError("changed collector; new source required")
        else:
            files = sorted((root / "output/tushare_cache/daily").glob("*.parquet"))
            if not files:
                raise ValueError("no daily references")
            plan = {"code_sha256": digest(Path(__file__)), "api": "suspend_d", "fields": FIELDS,
                    "dates": [{"date": p.stem, "sha256": digest(p)} for p in files]}
            atomic_json(plan_path, plan)
        for item in plan["dates"]:
            date = item["date"]
            if not re.fullmatch(r"\d{8}", date):
                raise ValueError("invalid date")
            if digest(root / "output/tushare_cache/daily" / (date + ".parquet")) != item["sha256"]:
                raise ValueError("daily reference changed")
            data_path, receipt_path = output / (date + ".parquet"), output / (date + ".json")
            if receipt_path.exists():
                receipt = load_plan(receipt_path)
                if receipt["date"] != date or digest(data_path) != receipt["sha256"]:
                    raise ValueError("resume receipt mismatch")
                check = inspect(pd.read_parquet(data_path), date)
                if check != receipt["check"]:
                    raise ValueError("cached schema check mismatch")
            else:
                if used >= max_requests:
                    break
                if data_path.exists():
                    raise ValueError("uncommitted raw data; no overwrite")
                requested = now()
                used += 1
                try:
                    frame = query(date)
                except Exception as exc:
                    atomic_json(output / ("failure-" + uuid.uuid4().hex + ".json"),
                                {"date": date, "at": now(), "error_type": type(exc).__name__})
                    raise RuntimeError("provider failure; sanitized evidence retained") from None
                received = now()
                frame.to_parquet(data_path, index=False)
                check = inspect(frame, date)
                receipt = {"date": date, "requested_at": requested, "received_at": received,
                           "sha256": digest(data_path), "check": check}
                atomic_json(receipt_path, receipt)
                if pause:
                    time.sleep(pause)
            if check["invalid_rows"] or check["duplicate_rows"] or check["possible_truncation"]:
                raise ValueError("invalid or potentially truncated response retained")
            records.append(receipt)
            heartbeat = dict(identity, source_id=source_id, completed_dates=len(records), total_dates=len(plan["dates"]), at=now())
            atomic_json(output / "heartbeat.json", heartbeat)
            if len(records) == 1 or len(records) % 25 == 0:
                print(json.dumps(heartbeat), flush=True)
        report = {"source_id": source_id, "completed_dates": len(records), "total_dates": len(plan["dates"]),
                  "complete_acquisition": len(records) == len(plan["dates"]), "requests_used": used,
                  "rows": sum(r["check"]["rows"] for r in records), "formal_H01_gate_passed": False,
                  "semantics": "retrospective provider rows; timing/null, coverage and historical availability need audit"}
        atomic_json(output / "summary.json", report)
        return report
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-id")
    parser.add_argument("--max-requests", type=int, default=700)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    from dotenv import load_dotenv
    import requests
    load_dotenv(root / ".env", override=False)
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise SystemExit("provider credential unavailable")
    session = requests.Session()
    def query(date):
        response = session.post("https://api.tushare.pro", json={"api_name": "suspend_d", "token": token,
                                "params": {"trade_date": date}, "fields": FIELDS}, timeout=25)
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise ValueError("provider rejected request")
        return pd.DataFrame(payload["data"]["items"], columns=payload["data"]["fields"])
    print(json.dumps(collect(root, query, args.source_id, args.max_requests), indent=2))


if __name__ == "__main__":
    main()
